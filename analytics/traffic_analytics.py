"""
TraceX SIH 2026 - Traffic Analytics Engine
Analyzes vehicle detection volume, camera load, hourly patterns, transit speeds, and route patterns.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


class TrafficAnalyticsEngine:
    """
    Core engine for computing traffic intelligence metrics from PostgreSQL + PostGIS.
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
        """Establish connection to PostgreSQL."""
        return psycopg2.connect(**self.db_params)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute high-level dataset metrics."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_detections,
                        COUNT(DISTINCT track_id) as unique_vehicles,
                        COUNT(DISTINCT cleaned_plate) FILTER (WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ') as unique_plates,
                        COUNT(*) FILTER (WHERE valid_indian_plate = TRUE) as valid_indian_plates,
                        COUNT(DISTINCT camera_id) as active_cameras,
                        MIN(timestamp) as earliest_detection,
                        MAX(timestamp) as latest_detection
                    FROM vehicle_detections;
                """)
                row = cur.fetchone() or {}

        total = row.get("total_detections") or 0
        valid_count = row.get("valid_indian_plates") or 0
        valid_pct = round((valid_count / total * 100), 2) if total > 0 else 0.0

        return {
            "total_detections": total,
            "unique_vehicles": row.get("unique_vehicles") or 0,
            "unique_plates": row.get("unique_plates") or 0,
            "valid_indian_plates": valid_count,
            "valid_indian_percentage": valid_pct,
            "active_cameras": row.get("active_cameras") or 0,
            "earliest_detection": str(row.get("earliest_detection")) if row.get("earliest_detection") else None,
            "latest_detection": str(row.get("latest_detection")) if row.get("latest_detection") else None,
        }

    def get_camera_traffic_stats(self) -> List[Dict[str, Any]]:
        """Compute per-camera vehicle flow and detection counts."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        camera_id,
                        COUNT(*) as detection_count,
                        COUNT(DISTINCT track_id) as vehicle_count,
                        COUNT(DISTINCT cleaned_plate) FILTER (WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ') as unique_plates_count,
                        AVG(yolo_confidence) as avg_yolo_conf,
                        AVG(ocr_confidence) as avg_ocr_conf,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_seen
                    FROM vehicle_detections
                    GROUP BY camera_id
                    ORDER BY detection_count DESC;
                """)
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "camera_id": r["camera_id"],
                "detection_count": r["detection_count"],
                "vehicle_count": r["vehicle_count"],
                "unique_plates_count": r["unique_plates_count"],
                "avg_yolo_confidence": round(float(r["avg_yolo_conf"] or 0.0), 3),
                "avg_ocr_confidence": round(float(r["avg_ocr_conf"] or 0.0), 3),
                "first_seen": str(r["first_seen"]) if r["first_seen"] else None,
                "last_seen": str(r["last_seen"]) if r["last_seen"] else None,
            })
        return results

    def get_hourly_traffic_distribution(self) -> List[Dict[str, Any]]:
        """Compute traffic volume distribution aggregated by hour of the day."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        EXTRACT(HOUR FROM timestamp)::INTEGER as hour_of_day,
                        COUNT(*) as detection_count,
                        COUNT(DISTINCT track_id) as vehicle_count,
                        COUNT(DISTINCT cleaned_plate) FILTER (WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ') as plate_count
                    FROM vehicle_detections
                    GROUP BY hour_of_day
                    ORDER BY hour_of_day ASC;
                """)
                rows = cur.fetchall()

        results = []
        for r in rows:
            h = r["hour_of_day"]
            results.append({
                "hour_of_day": h,
                "hour_label": f"{h:02d}:00 - {(h+1)%24:02d}:00",
                "detection_count": r["detection_count"],
                "vehicle_count": r["vehicle_count"],
                "plate_count": r["plate_count"],
            })
        return results

    def get_speed_and_distance_analytics(self) -> Dict[str, Any]:
        """
        Compute inter-camera transit metrics:
        - average, max, and min segment speeds (km/h)
        - average and max path distance (km)
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query segment distances and time deltas between consecutive camera observations
                cur.execute("""
                    WITH segment_data AS (
                        SELECT 
                            cleaned_plate,
                            camera_id,
                            timestamp,
                            ST_Distance(
                                location, 
                                LAG(location) OVER (PARTITION BY cleaned_plate ORDER BY timestamp ASC, frame_number ASC)
                            ) as leg_dist_meters,
                            EXTRACT(
                                EPOCH FROM (
                                    timestamp - LAG(timestamp) OVER (PARTITION BY cleaned_plate ORDER BY timestamp ASC, frame_number ASC)
                                )
                            ) as time_delta_sec
                        FROM vehicle_detections
                        WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ' AND location IS NOT NULL
                    ),
                    valid_transit_segments AS (
                        SELECT 
                            cleaned_plate,
                            leg_dist_meters,
                            time_delta_sec,
                            (leg_dist_meters / time_delta_sec) * 3.6 as speed_kmh
                        FROM segment_data
                        WHERE time_delta_sec > 0 AND leg_dist_meters > 0
                    )
                    SELECT 
                        COUNT(*) as total_transit_segments,
                        AVG(speed_kmh) as avg_speed_kmh,
                        MAX(speed_kmh) as max_speed_kmh,
                        MIN(speed_kmh) as min_speed_kmh,
                        AVG(leg_dist_meters) as avg_segment_dist_meters
                    FROM valid_transit_segments;
                """)
                speed_row = cur.fetchone() or {}

                # Query total path distances per vehicle
                cur.execute("""
                    WITH vehicle_paths AS (
                        SELECT 
                            cleaned_plate,
                            SUM(leg_dist_meters) as total_path_meters,
                            COUNT(*) as sightings_count
                        FROM (
                            SELECT 
                                cleaned_plate,
                                ST_Distance(
                                    location, 
                                    LAG(location) OVER (PARTITION BY cleaned_plate ORDER BY timestamp ASC, frame_number ASC)
                                ) as leg_dist_meters
                            FROM vehicle_detections
                            WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ' AND location IS NOT NULL
                        ) t
                        WHERE leg_dist_meters IS NOT NULL AND leg_dist_meters > 0
                        GROUP BY cleaned_plate
                    )
                    SELECT 
                        COUNT(*) as vehicles_with_path,
                        AVG(total_path_meters) as avg_path_meters,
                        MAX(total_path_meters) as max_path_meters
                    FROM vehicle_paths;
                """)
                path_row = cur.fetchone() or {}

        total_segments = speed_row.get("total_transit_segments") or 0
        avg_speed = round(float(speed_row.get("avg_speed_kmh") or 0.0), 2)
        max_speed = round(float(speed_row.get("max_speed_kmh") or 0.0), 2)
        min_speed = round(float(speed_row.get("min_speed_kmh") or 0.0), 2)

        vehicles_with_path = path_row.get("vehicles_with_path") or 0
        avg_path_km = round(float(path_row.get("avg_path_meters") or 0.0) / 1000.0, 2)
        max_path_km = round(float(path_row.get("max_path_meters") or 0.0) / 1000.0, 2)

        return {
            "total_transit_segments": total_segments,
            "average_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "min_speed_kmh": min_speed if total_segments > 0 else 0.0,
            "vehicles_with_multi_point_path": vehicles_with_path,
            "average_path_distance_km": avg_path_km,
            "max_path_distance_km": max_path_km,
        }

    def get_most_common_vehicle_routes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Identify and rank distinct multi-camera trajectory sequences."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH ordered_sightings AS (
                        SELECT 
                            cleaned_plate,
                            camera_id,
                            timestamp,
                            frame_number
                        FROM vehicle_detections
                        WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ'
                        ORDER BY cleaned_plate, timestamp ASC, frame_number ASC
                    ),
                    route_strings AS (
                        SELECT 
                            cleaned_plate,
                            STRING_AGG(DISTINCT camera_id, ' -> ' ORDER BY camera_id) as distinct_cameras_visited,
                            COUNT(DISTINCT camera_id) as unique_cameras_count,
                            MIN(timestamp) as journey_start,
                            MAX(timestamp) as journey_end
                        FROM ordered_sightings
                        GROUP BY cleaned_plate
                        HAVING COUNT(DISTINCT camera_id) > 1
                    )
                    SELECT 
                        distinct_cameras_visited as route,
                        COUNT(*) as vehicle_frequency,
                        ARRAY_AGG(cleaned_plate) as sample_plates
                    FROM route_strings
                    GROUP BY distinct_cameras_visited
                    ORDER BY vehicle_frequency DESC
                    LIMIT %s;
                """, (limit,))
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "route": r["route"],
                "frequency": r["vehicle_frequency"],
                "vehicles": r["sample_plates"],
            })
        return results

    def generate_full_analytics_report(self) -> Dict[str, Any]:
        """Generate complete analytics payload."""
        summary = self.get_summary_metrics()
        cameras = self.get_camera_traffic_stats()
        hourly = self.get_hourly_traffic_distribution()
        speed_dist = self.get_speed_and_distance_analytics()
        routes = self.get_most_common_vehicle_routes()

        return {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "cameras": cameras,
            "hourly_traffic": hourly,
            "speed_and_distance": speed_dist,
            "common_routes": routes,
        }

    def export_analytics_json(self, output_path: str = "data/traffic_analytics.json") -> str:
        """Export analytics metrics to formatted JSON file."""
        report = self.generate_full_analytics_report()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path
