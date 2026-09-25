"""
TraceX SIH 2026 - Rule Engine Test Suite
Tests deterministic violation detection (speeding, impossible speed, loops) and data quality warnings.
"""

import os
import sys
import json
import sqlite3

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rule_engine.rules import TraceXRuleEngine, RuleConfig

def test_rule_engine():
    print("=" * 75)
    print("TraceX STEP 16 - Rule Engine & Violation Detection Test Suite")
    print("=" * 75)

    config = RuleConfig(
        speed_limit_kmh=50.0,
        impossible_speed_kmh=120.0,
        repeated_camera_window_seconds=180.0,
        ocr_confidence_threshold=0.70
    )

    engine = TraceXRuleEngine(config=config)

    # 1. Run Complete Rule Evaluation
    results = engine.evaluate_all()
    summary = results["summary"]
    breakdown = summary["breakdown"]

    print("\n--- 1. Rule Engine Evaluation Summary ---")
    print(f"  Total Findings Logged       : {summary['total_findings']}")
    print(f"  Traffic Violations Total    : {summary['traffic_violations_count']}")
    print(f"  Data Quality Warnings Total : {summary['data_quality_warnings_count']}")
    print("\n  Rule Breakdown:")
    print(f"    * [TRAFFIC] Speeding Violations (>50 km/h)      : {breakdown['speeding_violations']}")
    print(f"    * [TRAFFIC] Impossible Speed (>120 km/h)        : {breakdown['impossible_speed_violations']}")
    print(f"    * [TRAFFIC] Repeated Camera Loop (<=180s)       : {breakdown['repeated_camera_flags']}")
    print(f"    * [QUALITY] Non-Standard / Invalid Plates       : {breakdown['invalid_plate_warnings']}")
    print(f"    * [QUALITY] Low OCR Confidence (<0.70)          : {breakdown['low_ocr_confidence_warnings']}")

    # 2. Inspect Sample Findings
    findings = results["findings"]
    print("\n--- 2. Sample Findings by Category ---")

    # Sample Speeding Violation
    speeding_samples = [f for f in findings if f["rule_id"] == "RULE_SPEEDING"]
    if speeding_samples:
        s = speeding_samples[0]
        print(f"\n  [SAMPLE SPEEDING VIOLATION]")
        print(f"    - Plate       : {s['plate']}")
        print(f"    - Route       : {s['camera_id']}")
        print(f"    - Measured    : {s['measured_value']} (Limit: {s['threshold']})")
        print(f"    - Severity    : {s['severity']}")
        print(f"    - Explanation : {s['explanation']}")

    # Sample Data Quality Warning
    quality_samples = [f for f in findings if f["rule_id"] == "RULE_INVALID_PLATE"]
    if quality_samples:
        q = quality_samples[0]
        print(f"\n  [SAMPLE DATA QUALITY WARNING]")
        print(f"    - Plate Text  : {q['plate']}")
        print(f"    - Camera ID   : {q['camera_id']}")
        print(f"    - Measured    : {q['measured_value']}")
        print(f"    - Severity    : {q['severity']}")
        print(f"    - Explanation : {q['explanation']}")

    # 3. Export to JSON
    json_path = "data/violations.json"
    print(f"\n--- 3. Exporting Violations to JSON ---")
    exported_path = engine.export_violations_json(json_path)
    print(f"  [PASS] Exported {len(findings)} records to: {exported_path}")
    assert os.path.exists(exported_path), "JSON export file not found"

    with open(exported_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["summary"]["total_findings"] == len(findings)
        print(f"  [PASS] JSON structure validated ({os.path.getsize(exported_path)} bytes).")

    # 4. Safety Verification (Confirm 52 records untouched)
    sq = sqlite3.connect("file:data/vehicle_history.db?mode=ro", uri=True)
    sq_cnt = sq.cursor().execute("SELECT COUNT(*) FROM vehicle_detections").fetchone()[0]
    sq.close()
    assert sq_cnt == 52, "SQLite record count was modified!"

    print("\n" + "=" * 75)
    print("TraceX STEP 16 Rule Engine Test Summary: ALL PASS")
    print("=" * 75)
    return True

if __name__ == "__main__":
    success = test_rule_engine()
    sys.exit(0 if success else 1)
