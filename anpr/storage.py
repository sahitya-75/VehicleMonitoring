import os
import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class VehicleDetectionRecord:
    """
    Standardized Vehicle Detection Record model for ANPR and Trajectory Reconstruction.
    """
    plate_text: str
    cleaned_plate: str
    yolo_confidence: float
    ocr_confidence: float
    valid_indian_plate: bool
    camera_id: str
    timestamp: str
    frame_number: int
    track_id: int
    latitude: float
    longitude: float
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        d = asdict(self)
        if d.get("id") is None:
            d.pop("id", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VehicleDetectionRecord":
        """Instantiate record from dictionary."""
        return cls(
            id=data.get("id"),
            plate_text=str(data.get("plate_text", "")),
            cleaned_plate=str(data.get("cleaned_plate", "")),
            yolo_confidence=float(data.get("yolo_confidence", 0.0)),
            ocr_confidence=float(data.get("ocr_confidence", 0.0)),
            valid_indian_plate=bool(data.get("valid_indian_plate", False)),
            camera_id=str(data.get("camera_id", "CAM_01")),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            frame_number=int(data.get("frame_number", 0)),
            track_id=int(data.get("track_id", 0)),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0))
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "VehicleDetectionRecord":
        """Instantiate record from SQLite Row."""
        return cls(
            id=row["id"],
            plate_text=row["plate_text"],
            cleaned_plate=row["cleaned_plate"],
            yolo_confidence=row["yolo_confidence"],
            ocr_confidence=row["ocr_confidence"],
            valid_indian_plate=bool(row["valid_indian_plate"]),
            camera_id=row["camera_id"],
            timestamp=row["timestamp"],
            frame_number=row["frame_number"],
            track_id=row["track_id"],
            latitude=row["latitude"],
            longitude=row["longitude"]
        )


class VehicleHistoryStore:
    """
    Local SQLite Storage Engine for Vehicle Detection History and Trajectory Tracking.
    Provides fast indexed queries, historical reconstruction, and JSON export.
    """

    def __init__(self, db_path: str = "data/vehicle_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create database tables and indices if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    cleaned_plate TEXT NOT NULL,
                    yolo_confidence REAL NOT NULL,
                    ocr_confidence REAL NOT NULL,
                    valid_indian_plate INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Indices for fast queries and trajectory reconstruction
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cleaned_plate ON vehicle_detections(cleaned_plate);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_detections(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_id ON vehicle_detections(track_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera_id ON vehicle_detections(camera_id);")
            conn.commit()

    def insert_record(self, record: VehicleDetectionRecord) -> int:
        """
        Insert a single vehicle detection record.
        Returns the inserted record's database ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vehicle_detections (
                    plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                    valid_indian_plate, camera_id, timestamp, frame_number,
                    track_id, latitude, longitude
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.plate_text,
                record.cleaned_plate,
                record.yolo_confidence,
                record.ocr_confidence,
                1 if record.valid_indian_plate else 0,
                record.camera_id,
                record.timestamp,
                record.frame_number,
                record.track_id,
                record.latitude,
                record.longitude
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_records_batch(self, records: List[VehicleDetectionRecord]) -> int:
        """
        Insert a batch of vehicle detection records efficiently.
        Returns total count of inserted records.
        """
        if not records:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = [
                (
                    r.plate_text,
                    r.cleaned_plate,
                    r.yolo_confidence,
                    r.ocr_confidence,
                    1 if r.valid_indian_plate else 0,
                    r.camera_id,
                    r.timestamp,
                    r.frame_number,
                    r.track_id,
                    r.latitude,
                    r.longitude
                ) for r in records
            ]
            cursor.executemany("""
                INSERT INTO vehicle_detections (
                    plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                    valid_indian_plate, camera_id, timestamp, frame_number,
                    track_id, latitude, longitude
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            return len(records)

    def get_history_by_plate(self, plate: str) -> List[VehicleDetectionRecord]:
        """
        Retrieve all chronological detections for a given license plate number.
        """
        cleaned = plate.strip().upper().replace(" ", "").replace("-", "")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM vehicle_detections
                WHERE cleaned_plate = ? OR cleaned_plate LIKE ?
                ORDER BY timestamp ASC, frame_number ASC
            """, (cleaned, f"%{cleaned}%"))
            rows = cursor.fetchall()
            return [VehicleDetectionRecord.from_row(row) for row in rows]

    def get_trajectory_by_plate(self, plate: str) -> List[Dict[str, Any]]:
        """
        Reconstruct chronological trajectory points for a given plate.
        Returns a list of GPS waypoints with camera IDs and timestamps.
        """
        history = self.get_history_by_plate(plate)
        trajectory = []
        for r in history:
            trajectory.append({
                "camera_id": r.camera_id,
                "timestamp": r.timestamp,
                "frame_number": r.frame_number,
                "track_id": r.track_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "ocr_confidence": r.ocr_confidence,
                "yolo_confidence": r.yolo_confidence
            })
        return trajectory

    def get_all_unique_plates(self) -> List[Dict[str, Any]]:
        """
        Retrieve list of all unique plate numbers observed, with occurrence counts and latest timestamp.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    cleaned_plate,
                    COUNT(*) as total_sightings,
                    COUNT(DISTINCT camera_id) as cameras_count,
                    COUNT(DISTINCT track_id) as tracks_count,
                    MAX(ocr_confidence) as max_ocr_conf,
                    MAX(valid_indian_plate) as is_valid,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM vehicle_detections
                WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ'
                GROUP BY cleaned_plate
                ORDER BY total_sightings DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_detections(self, limit: int = 50) -> List[VehicleDetectionRecord]:
        """
        Retrieve most recent detection records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM vehicle_detections
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [VehicleDetectionRecord.from_row(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics of the detection database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM vehicle_detections;")
            total_records = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(DISTINCT cleaned_plate) as unique_plates FROM vehicle_detections WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ';")
            unique_plates = cursor.fetchone()["unique_plates"]

            cursor.execute("SELECT COUNT(DISTINCT track_id) as unique_tracks FROM vehicle_detections;")
            unique_tracks = cursor.fetchone()["unique_tracks"]

            cursor.execute("SELECT COUNT(*) as valid_count FROM vehicle_detections WHERE valid_indian_plate = 1;")
            valid_plates = cursor.fetchone()["valid_count"]

            return {
                "total_detections": total_records,
                "unique_plates": unique_plates,
                "unique_tracks": unique_tracks,
                "valid_indian_plates": valid_plates,
                "db_path": self.db_path
            }

    def export_to_json(self, json_path: str = "data/vehicle_history.json") -> str:
        """
        Export complete vehicle history to a formatted JSON file.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicle_detections ORDER BY timestamp ASC, frame_number ASC;")
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]

        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)

        return json_path

    def clear(self):
        """Clear all records from database (useful for fresh benchmark runs)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vehicle_detections;")
            conn.commit()


# Module self-test
if __name__ == "__main__":
    store = VehicleHistoryStore("data/test_vehicle_history.db")
    rec = VehicleDetectionRecord(
        plate_text="HR51BC3493",
        cleaned_plate="HR51BC3493",
        yolo_confidence=0.88,
        ocr_confidence=0.94,
        valid_indian_plate=True,
        camera_id="CAM_01_JUNCTION_A",
        timestamp="2026-09-15T12:00:00.000Z",
        frame_number=15,
        track_id=1,
        latitude=28.6139,
        longitude=77.2090
    )
    rec_id = store.insert_record(rec)
    print(f"Inserted record ID: {rec_id}")
    history = store.get_history_by_plate("HR51BC3493")
    print(f"Queried history for HR51BC3493: {len(history)} records")
    print(f"Record details: {history[0].to_dict()}")
