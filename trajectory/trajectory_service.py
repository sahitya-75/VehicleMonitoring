"""
TraceX SIH 2026 - Trajectory Reconstruction Engine
Reconstructs historical spatial-temporal vehicle paths across camera sensor networks.
Supports exact plate matching and fuzzy OCR variation matching with configurable edit distance.
"""

import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    """
    s1, s2 = s1.upper().strip(), s2.upper().strip()
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def clean_plate_string(plate: str) -> str:
    """Standardize plate representation for search."""
    if not plate:
        return ""
    return "".join(c for c in plate.upper() if c.isalnum())


@dataclass
class TrajectoryPoint:
    """Represents a single verified spatial-temporal sighting of a vehicle."""
    id: int
    camera_id: str
    timestamp: str
    frame_number: int
    track_id: int
    latitude: float
    longitude: float
    plate_text: str
    cleaned_plate: str
    yolo_confidence: float
    ocr_confidence: float
    valid_indian_plate: bool
    distance_from_prev_m: float = 0.0
    time_delta_seconds: float = 0.0
    speed_kmh: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleTrajectory:
    """Represents an ordered chronological trajectory for a vehicle plate."""
    searched_plate: str
    matched_plate: str
    match_type: str  # 'EXACT' or 'FUZZY'
    edit_distance: int
    total_sightings: int
    first_seen: str
    last_seen: str
    total_distance_meters: float
    unique_cameras: List[str]
    points: List[TrajectoryPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d


class TrajectoryService:
    """
    Core Service for querying and reconstructing vehicle trajectories from PostgreSQL + PostGIS.
    """

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        load_dotenv()
        if db_params:
            self.db_params = db_params
        else:
            self.db_params = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "dbname": os.getenv("DB_NAME", "tracex"),
                "user": os.getenv("DB_USER", "tracex_user"),
                "password": os.getenv("DB_PASSWORD", "tracex_password_2026"),
                "connect_timeout": 5,
            }

    def _get_connection(self):
        """Establish a PostgreSQL connection."""
        return psycopg2.connect(**self.db_params)

    def get_all_unique_plates(self) -> List[Dict[str, Any]]:
        """Fetch summary of all unique plates observed in database."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        cleaned_plate,
                        COUNT(*) as total_sightings,
                        COUNT(DISTINCT camera_id) as cameras_count,
                        COUNT(DISTINCT track_id) as tracks_count,
                        MAX(ocr_confidence) as max_ocr_conf,
                        BOOL_OR(valid_indian_plate) as is_valid,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_seen
                    FROM vehicle_detections
                    WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ'
                    GROUP BY cleaned_plate
                    ORDER BY total_sightings DESC;
                """)
                return [dict(r) for r in cur.fetchall()]

    def reconstruct_trajectory_for_plate(
        self,
        searched_plate: str,
        target_plate: str,
        match_type: str,
        edit_dist: int
    ) -> Optional[VehicleTrajectory]:
        """
        Fetch all records for target_plate in chronological order and compute spatial trajectory points.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use PostGIS ST_Distance against lag(location) to compute geodesic distance on the fly
                cur.execute("""
                    SELECT 
                        id, plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                        valid_indian_plate, camera_id, timestamp, frame_number, track_id,
                        latitude, longitude,
                        COALESCE(
                            ST_Distance(
                                location, 
                                LAG(location) OVER (ORDER BY timestamp ASC, frame_number ASC)
                            ), 0.0
                        ) as dist_from_prev_m,
                        EXTRACT(
                            EPOCH FROM (
                                timestamp - LAG(timestamp) OVER (ORDER BY timestamp ASC, frame_number ASC)
                            )
                        ) as time_delta_sec
                    FROM vehicle_detections
                    WHERE cleaned_plate = %s
                    ORDER BY timestamp ASC, frame_number ASC;
                """, (target_plate,))
                rows = cur.fetchall()

        if not rows:
            return None

        points: List[TrajectoryPoint] = []
        total_distance = 0.0
        unique_cams = set()

        for r in rows:
            dist_m = float(r["dist_from_prev_m"] or 0.0)
            time_sec = float(r["time_delta_sec"] or 0.0)
            speed_kmh = 0.0
            if time_sec > 0:
                speed_kmh = (dist_m / time_sec) * 3.6  # m/s to km/h

            total_distance += dist_m
            unique_cams.add(r["camera_id"])

            ts_str = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])

            point = TrajectoryPoint(
                id=r["id"],
                camera_id=r["camera_id"],
                timestamp=ts_str,
                frame_number=r["frame_number"],
                track_id=r["track_id"],
                latitude=float(r["latitude"]) if r["latitude"] is not None else 0.0,
                longitude=float(r["longitude"]) if r["longitude"] is not None else 0.0,
                plate_text=r["plate_text"],
                cleaned_plate=r["cleaned_plate"],
                yolo_confidence=float(r["yolo_confidence"]),
                ocr_confidence=float(r["ocr_confidence"]),
                valid_indian_plate=bool(r["valid_indian_plate"]),
                distance_from_prev_m=round(dist_m, 2),
                time_delta_seconds=round(time_sec, 2),
                speed_kmh=round(speed_kmh, 2)
            )
            points.append(point)

        first_ts = points[0].timestamp
        last_ts = points[-1].timestamp

        return VehicleTrajectory(
            searched_plate=searched_plate,
            matched_plate=target_plate,
            match_type=match_type,
            edit_distance=edit_dist,
            total_sightings=len(points),
            first_seen=first_ts,
            last_seen=last_ts,
            total_distance_meters=round(total_distance, 2),
            unique_cameras=sorted(list(unique_cams)),
            points=points
        )

    def get_exact_trajectory(self, plate: str) -> Optional[VehicleTrajectory]:
        """
        Reconstruct trajectory for exact cleaned plate match.
        """
        cleaned = clean_plate_string(plate)
        if not cleaned:
            return None
        return self.reconstruct_trajectory_for_plate(
            searched_plate=cleaned,
            target_plate=cleaned,
            match_type="EXACT",
            edit_dist=0
        )

    def query_trajectory(
        self,
        plate: str,
        allow_fuzzy: bool = True,
        max_edit_distance: int = 2
    ) -> List[VehicleTrajectory]:
        """
        Query vehicle trajectory.
        1. Checks for exact match.
        2. If allow_fuzzy is True, checks all database plates for edit distance <= max_edit_distance.
        Returns list of matched trajectories grouped by matched vehicle plate, sorted by relevance.
        """
        cleaned_search = clean_plate_string(plate)
        if not cleaned_search:
            return []

        trajectories: List[VehicleTrajectory] = []

        # 1. Try Exact match
        exact_traj = self.get_exact_trajectory(cleaned_search)
        if exact_traj:
            trajectories.append(exact_traj)

        # 2. Fuzzy search if requested
        if allow_fuzzy:
            unique_plates_summary = self.get_all_unique_plates()
            for row in unique_plates_summary:
                cand_plate = row["cleaned_plate"]
                if cand_plate == cleaned_search:
                    continue  # already added in exact match

                dist = levenshtein_distance(cleaned_search, cand_plate)
                if dist <= max_edit_distance:
                    fuzz_traj = self.reconstruct_trajectory_for_plate(
                        searched_plate=cleaned_search,
                        target_plate=cand_plate,
                        match_type="FUZZY",
                        edit_dist=dist
                    )
                    if fuzz_traj:
                        trajectories.append(fuzz_traj)

        # Sort trajectories: exact first (edit_dist=0), then ascending edit_dist, then descending sightings
        trajectories.sort(key=lambda t: (t.edit_distance, -t.total_sightings))
        return trajectories
