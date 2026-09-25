"""
TraceX SIH 2026 - Alerts & Watchlist Test Suite
Tests watchlist management, real-time detection matching, rule-based alerts, and duplicate prevention.
"""

import os
import sys
import json
import sqlite3

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alerts.alert_service import AlertService, WatchlistItem, AlertRecord

def test_alert_service():
    print("=" * 75)
    print("TraceX STEP 17 - Alerts & Watchlist Test Suite")
    print("=" * 75)

    watchlist_path = "data/watchlist.json"
    alerts_path = "data/alerts.json"

    service = AlertService(
        watchlist_file=watchlist_path,
        alerts_file=alerts_path
    )
    service.clear_alerts()

    # 1. Test Watchlist Management
    print("\n--- 1. Testing Watchlist Operations ---")
    test_plate = "HR51BC3493"
    watch_item = service.add_to_watchlist(
        plate=test_plate,
        description="Stolen Vehicle - Case Ref #2026-DL-891",
        priority="CRITICAL",
        enabled=True
    )
    print(f"  [PASS] Added to Watchlist: {watch_item.cleaned_plate} | Priority: {watch_item.priority} | Reason: {watch_item.description}")

    watchlist_items = service.list_watchlist()
    assert len(watchlist_items) >= 1, "Watchlist should contain at least 1 item"
    assert any(w.cleaned_plate == test_plate for w in watchlist_items)
    print(f"  [PASS] Watchlist contains {len(watchlist_items)} active entry/entries.")

    # 2. Test Watchlist Scanning on Existing Detections
    print(f"\n--- 2. Scanning Database for Watchlist Matches ({test_plate}) ---")
    watchlist_alerts = service.scan_watchlist_hits()
    print(f"  Generated {len(watchlist_alerts)} Watchlist Alerts for '{test_plate}':")
    for idx, a in enumerate(watchlist_alerts, 1):
        print(f"    {idx}. [{a.alert_id}] {a.alert_type} | Severity: {a.severity} | Station: {a.camera_id} | Time: {a.timestamp}")

    assert len(watchlist_alerts) == 4, f"Expected 4 watchlist alerts for HR51BC3493, got {len(watchlist_alerts)}"

    # 3. Test Rule Engine Alert Processing (Speeding & Repeated Camera)
    print("\n--- 3. Processing Rule-Based Alerts (Speeding & Repeated Camera) ---")
    rule_alerts = service.process_rule_findings()
    speeding_alerts = [a for a in rule_alerts if a.alert_type == "SPEEDING_VIOLATION"]
    loop_alerts = [a for a in rule_alerts if a.alert_type == "REPEATED_CAMERA_ALERT"]

    print(f"  Total Rule-Based Alerts Generated : {len(rule_alerts)}")
    print(f"    * Speeding Violations (>50 km/h) : {len(speeding_alerts)}")
    print(f"    * Repeated Camera / Short Loops  : {len(loop_alerts)}")

    for idx, a in enumerate(speeding_alerts, 1):
        print(f"    {idx}. [{a.alert_id}] {a.plate} | {a.alert_type} ({a.severity}) | Camera: {a.camera_id}")

    assert len(speeding_alerts) == 4, f"Expected 4 speeding alerts, got {len(speeding_alerts)}"

    # 4. Test Duplicate Alert Prevention
    print("\n--- 4. Testing Duplicate Prevention ---")
    total_before = len(service.get_all_alerts())
    print(f"  Alerts count before re-scan: {total_before}")

    dup_watch = service.scan_watchlist_hits()
    dup_rules = service.process_rule_findings()
    total_after = len(service.get_all_alerts())

    print(f"  Re-scan watchlist new alerts: {len(dup_watch)}")
    print(f"  Re-scan rules new alerts    : {len(dup_rules)}")
    print(f"  Alerts count after re-scan : {total_after}")

    assert len(dup_watch) == 0, f"Duplicate watchlist alerts generated! Count: {len(dup_watch)}"
    assert len(dup_rules) == 0, f"Duplicate rule alerts generated! Count: {len(dup_rules)}"
    assert total_before == total_after, "Alerts total changed on re-scan!"
    print("  [PASS] Duplicate Prevention Verified: 0 duplicates created.")

    # 5. Verify Persistent JSON Files
    print("\n--- 5. Verifying Persistent JSON Storage ---")
    assert os.path.exists(watchlist_path), f"File {watchlist_path} not found"
    assert os.path.exists(alerts_path), f"File {alerts_path} not found"

    with open(alerts_path, "r", encoding="utf-8") as f:
        stored_alerts = json.load(f)
        assert len(stored_alerts) == total_after
        print(f"  [PASS] Stored alerts in {alerts_path}: {len(stored_alerts)} records ({os.path.getsize(alerts_path)} bytes).")

    with open(watchlist_path, "r", encoding="utf-8") as f:
        stored_watch = json.load(f)
        assert len(stored_watch) >= 1
        print(f"  [PASS] Stored watchlist in {watchlist_path}: {len(stored_watch)} entries ({os.path.getsize(watchlist_path)} bytes).")

    # 6. Source Data Integrity Verification
    sq = sqlite3.connect("file:data/vehicle_history.db?mode=ro", uri=True)
    sq_cnt = sq.cursor().execute("SELECT COUNT(*) FROM vehicle_detections").fetchone()[0]
    sq.close()
    assert sq_cnt == 52, "SQLite record count was modified!"

    print("\n" + "=" * 75)
    print("TraceX STEP 17 Alerts & Watchlist Test Summary: ALL PASS")
    print("=" * 75)
    return True

if __name__ == "__main__":
    success = test_alert_service()
    sys.exit(0 if success else 1)
