"""
TraceX SIH 2026 - Alerts & Watchlist Service
Manages target vehicle watchlists, performs real-time detection matching, and converts rule violations into alerts.
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Ensure project root in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rule_engine.rules import TraceXRuleEngine, ViolationRecord, RuleCategory, RuleSeverity


@dataclass
class WatchlistItem:
    """Represents a monitored vehicle in the watchlist."""
    cleaned_plate: str
    description: str
    priority: str = "HIGH"  # LOW, MEDIUM, HIGH, CRITICAL
    enabled: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.cleaned_plate = self.cleaned_plate.upper().strip().replace(" ", "").replace("-", "")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertRecord:
    """Represents an actionable alert generated from watchlist hits or rule violations."""
    alert_id: str
    plate: str
    alert_type: str  # WATCHLIST_HIT, SPEEDING_VIOLATION, IMPOSSIBLE_SPEED, REPEATED_CAMERA
    severity: str    # LOW, MEDIUM, HIGH, CRITICAL
    camera_id: str
    timestamp: str
    message: str
    source_rule_id: Optional[str] = None
    detection_id: Optional[int] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def deduplication_key(self) -> str:
        """Unique signature to prevent duplicate alert generation."""
        components = [
            str(self.plate).upper().strip(),
            str(self.alert_type),
            str(self.camera_id),
            str(self.timestamp),
            str(self.source_rule_id or "")
        ]
        return "|".join(components)


class AlertService:
    """
    Manages vehicle watchlists, detection matching, rule alert generation, and persistent JSON stores.
    """

    def __init__(
        self,
        watchlist_file: str = "data/watchlist.json",
        alerts_file: str = "data/alerts.json",
        db_params: Optional[Dict[str, Any]] = None
    ):
        load_dotenv()
        self.watchlist_file = watchlist_file
        self.alerts_file = alerts_file
        self.watchlist: Dict[str, WatchlistItem] = {}
        self.alerts: List[AlertRecord] = []
        self._alert_signatures: Set[str] = set()

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

        self._load_watchlist()
        self._load_alerts()

    def _get_connection(self):
        return psycopg2.connect(**self.db_params)

    # ------------------ Persistence ------------------

    def _load_watchlist(self):
        """Load watchlist items from JSON file."""
        if os.path.exists(self.watchlist_file):
            try:
                with open(self.watchlist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.watchlist = {
                        item["cleaned_plate"]: WatchlistItem(**item)
                        for item in data
                    }
            except Exception:
                self.watchlist = {}
        else:
            self.watchlist = {}

    def _save_watchlist(self):
        """Save watchlist items to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.watchlist_file)), exist_ok=True)
        with open(self.watchlist_file, "w", encoding="utf-8") as f:
            data = [item.to_dict() for item in self.watchlist.values()]
            json.dump(data, f, indent=2)

    def _load_alerts(self):
        """Load stored alerts and initialize deduplication signatures."""
        self.alerts = []
        self._alert_signatures = set()
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        alert = AlertRecord(**item)
                        self.alerts.append(alert)
                        self._alert_signatures.add(alert.deduplication_key)
            except Exception:
                self.alerts = []
                self._alert_signatures = set()

    def _save_alerts(self):
        """Save alerts to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.alerts_file)), exist_ok=True)
        with open(self.alerts_file, "w", encoding="utf-8") as f:
            data = [a.to_dict() for a in self.alerts]
            json.dump(data, f, indent=2)

    # ------------------ Watchlist Management ------------------

    def add_to_watchlist(
        self,
        plate: str,
        description: str,
        priority: str = "HIGH",
        enabled: bool = True
    ) -> WatchlistItem:
        """Add or update a vehicle on the watchlist."""
        cleaned = plate.upper().strip().replace(" ", "").replace("-", "")
        item = WatchlistItem(
            cleaned_plate=cleaned,
            description=description,
            priority=priority,
            enabled=enabled
        )
        self.watchlist[cleaned] = item
        self._save_watchlist()
        return item

    def remove_from_watchlist(self, plate: str) -> bool:
        """Remove a vehicle from the watchlist."""
        cleaned = plate.upper().strip().replace(" ", "").replace("-", "")
        if cleaned in self.watchlist:
            del self.watchlist[cleaned]
            self._save_watchlist()
            return True
        return False

    def list_watchlist(self, only_enabled: bool = False) -> List[WatchlistItem]:
        """List all watchlist entries."""
        if only_enabled:
            return [item for item in self.watchlist.values() if item.enabled]
        return list(self.watchlist.values())

    # ------------------ Alert Generation & Matching ------------------

    def _generate_alert_id(self, prefix: str = "ALT") -> str:
        """Generate unique, sequential alert identifier."""
        return f"{prefix}-{len(self.alerts) + 1:04d}"

    def scan_watchlist_hits(self) -> List[AlertRecord]:
        """
        Scan all detections in the database for active watchlist matches.
        Prevents duplicate alerts.
        """
        active_watch = {k: v for k, v in self.watchlist.items() if v.enabled}
        if not active_watch:
            return []

        new_alerts: List[AlertRecord] = []

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, plate_text, cleaned_plate, camera_id, timestamp, latitude, longitude
                    FROM vehicle_detections
                    WHERE cleaned_plate = ANY(%s)
                    ORDER BY timestamp ASC;
                """, (list(active_watch.keys()),))
                detections = cur.fetchall()

        for det in detections:
            plate = det["cleaned_plate"]
            watch_entry = active_watch.get(plate)
            if not watch_entry:
                continue

            ts_str = det["timestamp"].isoformat() if hasattr(det["timestamp"], "isoformat") else str(det["timestamp"])
            cam = det["camera_id"]

            temp_alert = AlertRecord(
                alert_id="",
                plate=plate,
                alert_type="WATCHLIST_HIT",
                severity=watch_entry.priority,
                camera_id=cam,
                timestamp=ts_str,
                message=f"Watchlist Hit: Vehicle '{plate}' detected at {cam}. Reason: {watch_entry.description}",
                source_rule_id="WATCHLIST_MONITOR",
                detection_id=det["id"]
            )

            # Deduplication Check
            if temp_alert.deduplication_key in self._alert_signatures:
                continue

            temp_alert.alert_id = self._generate_alert_id("ALT-WATCH")
            self.alerts.append(temp_alert)
            self._alert_signatures.add(temp_alert.deduplication_key)
            new_alerts.append(temp_alert)

        if new_alerts:
            self._save_alerts()

        return new_alerts

    def process_rule_findings(
        self,
        findings: Optional[List[ViolationRecord]] = None
    ) -> List[AlertRecord]:
        """
        Process rule engine findings and generate high-priority alerts for traffic violations
        (Speeding, Impossible Speed, Repeated Camera).
        """
        if findings is None:
            engine = TraceXRuleEngine(db_params=self.db_params)
            eval_res = engine.evaluate_all()
            findings = [ViolationRecord(**f) for f in eval_res["findings"]]

        new_alerts: List[AlertRecord] = []

        # High priority violation rules
        alert_eligible_rules = {
            "RULE_SPEEDING": "SPEEDING_VIOLATION",
            "RULE_IMPOSSIBLE_SPEED": "IMPOSSIBLE_SPEED_ALERT",
            "RULE_REPEATED_CAMERA": "REPEATED_CAMERA_ALERT"
        }

        for f in findings:
            if f.rule_id not in alert_eligible_rules:
                continue

            alert_type = alert_eligible_rules[f.rule_id]

            temp_alert = AlertRecord(
                alert_id="",
                plate=f.plate,
                alert_type=alert_type,
                severity=f.severity,
                camera_id=f.camera_id,
                timestamp=f.timestamp,
                message=f"[{f.rule_name}] {f.explanation}",
                source_rule_id=f.rule_id,
                detection_id=f.detection_id
            )

            # Deduplication Check
            if temp_alert.deduplication_key in self._alert_signatures:
                continue

            temp_alert.alert_id = self._generate_alert_id("ALT-RULE")
            self.alerts.append(temp_alert)
            self._alert_signatures.add(temp_alert.deduplication_key)
            new_alerts.append(temp_alert)

        if new_alerts:
            self._save_alerts()

        return new_alerts

    def get_all_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        plate: Optional[str] = None
    ) -> List[AlertRecord]:
        """Query and filter stored alerts."""
        results = self.alerts
        if alert_type:
            results = [a for a in results if a.alert_type == alert_type]
        if severity:
            results = [a for a in results if a.severity == severity]
        if plate:
            cleaned = plate.upper().strip().replace(" ", "").replace("-", "")
            results = [a for a in results if a.plate == cleaned]
        return results

    def clear_alerts(self):
        """Clear alerts store."""
        self.alerts = []
        self._alert_signatures = set()
        self._save_alerts()
