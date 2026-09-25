"""
TraceX SIH 2026 - Rule Engine & Violation Detection
Deterministic, explainable rule engine for detecting traffic infractions and data quality warnings.
"""

import os
import sys
import json
from enum import Enum
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


class RuleCategory(str, Enum):
    TRAFFIC_VIOLATION = "TRAFFIC_VIOLATION"
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"


class RuleSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    WARNING = "WARNING"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RuleConfig:
    """Configurable thresholds for rule evaluation."""
    speed_limit_kmh: float = 50.0
    impossible_speed_kmh: float = 120.0
    repeated_camera_window_seconds: float = 180.0  # 3 minutes
    ocr_confidence_threshold: float = 0.70


@dataclass
class ViolationRecord:
    """Standardized finding record produced by the rule engine."""
    rule_id: str
    rule_name: str
    category: str
    severity: str
    plate: str
    camera_id: str
    timestamp: str
    measured_value: str
    threshold: str
    explanation: str
    detection_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceXRuleEngine:
    """
    Evaluates detection and trajectory records against deterministic, explainable rules.
    """

    def __init__(
        self,
        config: Optional[RuleConfig] = None,
        db_params: Optional[Dict[str, Any]] = None
    ):
        load_dotenv()
        self.config = config or RuleConfig()
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
        return psycopg2.connect(**self.db_params)

    def check_speed_violations(self) -> List[ViolationRecord]:
        """
        Rule 1: SPEEDING (> speed_limit_kmh)
        Rule 2: IMPOSSIBLE SPEED (> impossible_speed_kmh)
        Evaluates inter-camera segment travel speeds.
        """
        findings: List[ViolationRecord] = []

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH segment_data AS (
                        SELECT 
                            id,
                            cleaned_plate,
                            camera_id,
                            timestamp,
                            frame_number,
                            LAG(camera_id) OVER (PARTITION BY cleaned_plate ORDER BY timestamp ASC, frame_number ASC) as prev_camera,
                            LAG(timestamp) OVER (PARTITION BY cleaned_plate ORDER BY timestamp ASC, frame_number ASC) as prev_timestamp,
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
                    )
                    SELECT 
                        id,
                        cleaned_plate,
                        prev_camera,
                        camera_id,
                        timestamp,
                        prev_timestamp,
                        leg_dist_meters,
                        time_delta_sec,
                        (leg_dist_meters / time_delta_sec) * 3.6 as speed_kmh
                    FROM segment_data
                    WHERE time_delta_sec > 0 AND leg_dist_meters > 0
                    ORDER BY timestamp ASC;
                """)
                segments = cur.fetchall()

        for seg in segments:
            speed = float(seg["speed_kmh"])
            dist_km = float(seg["leg_dist_meters"]) / 1000.0
            time_min = float(seg["time_delta_sec"]) / 60.0
            route_str = f"{seg['prev_camera']} -> {seg['camera_id']}"
            ts_str = seg["timestamp"].isoformat() if hasattr(seg["timestamp"], "isoformat") else str(seg["timestamp"])
            plate = seg["cleaned_plate"]
            rec_id = seg["id"]

            # Rule 2: Impossible Speed
            if speed > self.config.impossible_speed_kmh:
                findings.append(ViolationRecord(
                    rule_id="RULE_IMPOSSIBLE_SPEED",
                    rule_name="Impossible Transit Speed / Clone Anomaly",
                    category=RuleCategory.TRAFFIC_VIOLATION.value,
                    severity=RuleSeverity.CRITICAL.value,
                    plate=plate,
                    camera_id=route_str,
                    timestamp=ts_str,
                    measured_value=f"{speed:.2f} km/h",
                    threshold=f"{self.config.impossible_speed_kmh:.1f} km/h",
                    explanation=(
                        f"Vehicle traversed {dist_km:.2f} km in {time_min:.1f} min ({speed:.2f} km/h), "
                        f"which exceeds the physically plausible threshold ({self.config.impossible_speed_kmh:.1f} km/h). "
                        f"Possible cloned plate or timestamp anomaly."
                    ),
                    detection_id=rec_id
                ))
            # Rule 1: Speeding
            elif speed > self.config.speed_limit_kmh:
                excess = speed - self.config.speed_limit_kmh
                severity = RuleSeverity.HIGH.value if excess > 20 else RuleSeverity.MEDIUM.value
                findings.append(ViolationRecord(
                    rule_id="RULE_SPEEDING",
                    rule_name="Speed Limit Violation",
                    category=RuleCategory.TRAFFIC_VIOLATION.value,
                    severity=severity,
                    plate=plate,
                    camera_id=route_str,
                    timestamp=ts_str,
                    measured_value=f"{speed:.2f} km/h",
                    threshold=f"{self.config.speed_limit_kmh:.1f} km/h",
                    explanation=(
                        f"Vehicle traveled between {seg['prev_camera']} and {seg['camera_id']} "
                        f"({dist_km:.2f} km in {time_min:.1f} min) at an average speed of {speed:.2f} km/h, "
                        f"exceeding the speed limit of {self.config.speed_limit_kmh:.1f} km/h by {excess:.2f} km/h."
                    ),
                    detection_id=rec_id
                ))

        return findings

    def check_repeated_camera_loops(self) -> List[ViolationRecord]:
        """
        Rule 3: REPEATED CAMERA / LOOP
        Detects a vehicle re-appearing at the exact same camera station within repeated_camera_window_seconds.
        """
        findings: List[ViolationRecord] = []

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH consecutive_same_cam AS (
                        SELECT 
                            id,
                            cleaned_plate,
                            camera_id,
                            timestamp,
                            frame_number,
                            LAG(timestamp) OVER (PARTITION BY cleaned_plate, camera_id ORDER BY timestamp ASC, frame_number ASC) as prev_ts,
                            LAG(frame_number) OVER (PARTITION BY cleaned_plate, camera_id ORDER BY timestamp ASC, frame_number ASC) as prev_frame,
                            EXTRACT(
                                EPOCH FROM (
                                    timestamp - LAG(timestamp) OVER (PARTITION BY cleaned_plate, camera_id ORDER BY timestamp ASC, frame_number ASC)
                                )
                            ) as time_diff_sec
                        FROM vehicle_detections
                        WHERE cleaned_plate != '' AND cleaned_plate != 'NOT READ'
                    )
                    SELECT * 
                    FROM consecutive_same_cam
                    WHERE time_diff_sec IS NOT NULL AND time_diff_sec > 0 AND time_diff_sec <= %s
                    ORDER BY timestamp ASC;
                """, (self.config.repeated_camera_window_seconds,))
                rows = cur.fetchall()

        for r in rows:
            time_sec = float(r["time_diff_sec"])
            ts_str = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])
            findings.append(ViolationRecord(
                rule_id="RULE_REPEATED_CAMERA",
                rule_name="Repeated Camera Sighting / Short Loop",
                category=RuleCategory.TRAFFIC_VIOLATION.value,
                severity=RuleSeverity.LOW.value,
                plate=r["cleaned_plate"],
                camera_id=r["camera_id"],
                timestamp=ts_str,
                measured_value=f"{time_sec:.1f} seconds interval",
                threshold=f"<= {self.config.repeated_camera_window_seconds:.0f} seconds",
                explanation=(
                    f"Vehicle '{r['cleaned_plate']}' re-appeared at station {r['camera_id']} "
                    f"within {time_sec:.1f} seconds, indicating potential loitering or short circulatory movement."
                ),
                detection_id=r["id"]
            ))

        return findings

    def check_data_quality_warnings(self) -> List[ViolationRecord]:
        """
        Rule 4: INVALID / UNREADABLE PLATE (valid_indian_plate = False)
        Rule 5: LOW OCR CONFIDENCE (< ocr_confidence_threshold)
        Classified strictly as DATA QUALITY WARNINGS.
        """
        findings: List[ViolationRecord] = []

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        id,
                        plate_text,
                        cleaned_plate,
                        yolo_confidence,
                        ocr_confidence,
                        valid_indian_plate,
                        camera_id,
                        timestamp
                    FROM vehicle_detections
                    ORDER BY id ASC;
                """)
                rows = cur.fetchall()

        for r in rows:
            rec_id = r["id"]
            plate = r["cleaned_plate"] or r["plate_text"] or "UNREAD"
            cam = r["camera_id"]
            ts_str = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])
            ocr_conf = float(r["ocr_confidence"])
            is_valid = bool(r["valid_indian_plate"])

            # Rule 4: Invalid / Non-standard Indian plate format
            if not is_valid:
                findings.append(ViolationRecord(
                    rule_id="RULE_INVALID_PLATE",
                    rule_name="Non-Standard Plate Format",
                    category=RuleCategory.DATA_QUALITY_WARNING.value,
                    severity=RuleSeverity.WARNING.value,
                    plate=plate,
                    camera_id=cam,
                    timestamp=ts_str,
                    measured_value=f"Format Match: False ('{r['plate_text']}')",
                    threshold="Standard Indian Registration (e.g. DL01AB1234)",
                    explanation=(
                        f"Detections with text '{r['plate_text']}' did not match valid MoRTH Indian license "
                        f"plate syntax rules. Flagged for sensor review."
                    ),
                    detection_id=rec_id
                ))

            # Rule 5: Low OCR confidence
            if ocr_conf < self.config.ocr_confidence_threshold:
                findings.append(ViolationRecord(
                    rule_id="RULE_LOW_OCR_CONF",
                    rule_name="Low OCR Confidence Sighting",
                    category=RuleCategory.DATA_QUALITY_WARNING.value,
                    severity=RuleSeverity.LOW.value,
                    plate=plate,
                    camera_id=cam,
                    timestamp=ts_str,
                    measured_value=f"{ocr_conf * 100:.1f}%",
                    threshold=f">= {self.config.ocr_confidence_threshold * 100:.0f}%",
                    explanation=(
                        f"Optical character recognition confidence ({ocr_conf * 100:.1f}%) fell below "
                        f"quality threshold ({self.config.ocr_confidence_threshold * 100:.0f}%)."
                    ),
                    detection_id=rec_id
                ))

        return findings

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Execute all rules and aggregate results into categorized metrics and itemized list.
        """
        speed_findings = self.check_speed_violations()
        loop_findings = self.check_repeated_camera_loops()
        quality_findings = self.check_data_quality_warnings()

        all_findings = speed_findings + loop_findings + quality_findings

        # Aggregate summary counts
        speeding_count = sum(1 for f in all_findings if f.rule_id == "RULE_SPEEDING")
        impossible_speed_count = sum(1 for f in all_findings if f.rule_id == "RULE_IMPOSSIBLE_SPEED")
        repeated_camera_count = sum(1 for f in all_findings if f.rule_id == "RULE_REPEATED_CAMERA")
        invalid_plate_count = sum(1 for f in all_findings if f.rule_id == "RULE_INVALID_PLATE")
        low_ocr_count = sum(1 for f in all_findings if f.rule_id == "RULE_LOW_OCR_CONF")

        traffic_violations_total = speeding_count + impossible_speed_count + repeated_camera_count
        data_quality_warnings_total = invalid_plate_count + low_ocr_count

        return {
            "evaluated_at": datetime.now().isoformat(),
            "config": asdict(self.config),
            "summary": {
                "total_findings": len(all_findings),
                "traffic_violations_count": traffic_violations_total,
                "data_quality_warnings_count": data_quality_warnings_total,
                "breakdown": {
                    "speeding_violations": speeding_count,
                    "impossible_speed_violations": impossible_speed_count,
                    "repeated_camera_flags": repeated_camera_count,
                    "invalid_plate_warnings": invalid_plate_count,
                    "low_ocr_confidence_warnings": low_ocr_count,
                }
            },
            "findings": [f.to_dict() for f in all_findings]
        }

    def export_violations_json(self, output_path: str = "data/violations.json") -> str:
        """Export full violation evaluation to JSON file."""
        report = self.evaluate_all()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path
