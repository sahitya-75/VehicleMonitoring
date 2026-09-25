"""
TraceX SIH 2026 - Traffic Analytics Test Suite
Tests traffic volume, camera statistics, hourly distributions, transit speeds, and route metrics.
"""

import os
import sys
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.traffic_analytics import TrafficAnalyticsEngine

def test_traffic_analytics():
    print("=" * 75)
    print("TraceX STEP 15 - Traffic Analytics Engine Verification")
    print("=" * 75)

    engine = TrafficAnalyticsEngine()

    # 1. Summary Metrics
    print("\n--- 1. Summary Metrics ---")
    summary = engine.get_summary_metrics()
    print(f"  Total Detections        : {summary['total_detections']}")
    print(f"  Unique Vehicles (Tracks): {summary['unique_vehicles']}")
    print(f"  Unique Plates           : {summary['unique_plates']}")
    print(f"  Valid Indian Plates     : {summary['valid_indian_plates']} ({summary['valid_indian_percentage']}%)")
    print(f"  Active Camera Nodes     : {summary['active_cameras']}")
    print(f"  Time Window             : {summary['earliest_detection']} -> {summary['latest_detection']}")

    assert summary["total_detections"] == 52, f"Expected 52 total detections, got {summary['total_detections']}"
    assert summary["unique_vehicles"] > 0, "Expected unique vehicles > 0"
    assert summary["unique_plates"] > 0, "Expected unique plates > 0"

    # 2. Camera Traffic Distribution
    print("\n--- 2. Camera Node Traffic Breakdown ---")
    cameras = engine.get_camera_traffic_stats()
    print(f"  {'Camera ID':<26} {'Detections':<12} {'Vehicles':<10} {'Plates':<10} {'Avg YOLO':<10} {'Avg OCR':<10}")
    print("  " + "-" * 78)
    for c in cameras:
        print(f"  {c['camera_id']:<26} {c['detection_count']:<12} {c['vehicle_count']:<10} {c['unique_plates_count']:<10} {c['avg_yolo_confidence']:<10.3f} {c['avg_ocr_confidence']:<10.3f}")

    assert len(cameras) >= 5, f"Expected at least 5 camera stations, got {len(cameras)}"

    # 3. Hourly Traffic Volume
    print("\n--- 3. Hourly Traffic Distribution ---")
    hourly = engine.get_hourly_traffic_distribution()
    print(f"  {'Time Window':<18} {'Detections':<12} {'Vehicles':<10} {'Plates':<10}")
    print("  " + "-" * 52)
    for h in hourly:
        print(f"  {h['hour_label']:<18} {h['detection_count']:<12} {h['vehicle_count']:<10} {h['plate_count']:<10}")

    assert len(hourly) > 0, "Expected hourly distribution rows"

    # 4. Speed & Distance Analytics
    print("\n--- 4. Transit Speed & Trip Distance Analytics ---")
    speed_dist = engine.get_speed_and_distance_analytics()
    print(f"  Transit Segments Analyzed  : {speed_dist['total_transit_segments']}")
    print(f"  Average Travel Speed       : {speed_dist['average_speed_kmh']:.2f} km/h")
    print(f"  Maximum Observed Speed     : {speed_dist['max_speed_kmh']:.2f} km/h")
    print(f"  Minimum Observed Speed     : {speed_dist['min_speed_kmh']:.2f} km/h")
    print(f"  Vehicles with Multi-Node Route: {speed_dist['vehicles_with_multi_point_path']}")
    print(f"  Average Path Distance      : {speed_dist['average_path_distance_km']:.2f} km")
    print(f"  Maximum Path Distance      : {speed_dist['max_path_distance_km']:.2f} km")

    assert speed_dist["total_transit_segments"] == 12, f"Expected 12 transit segments, got {speed_dist['total_transit_segments']}"
    assert speed_dist["vehicles_with_multi_point_path"] == 4, f"Expected 4 vehicles with paths, got {speed_dist['vehicles_with_multi_point_path']}"
    assert speed_dist["max_speed_kmh"] > 0, "Expected max speed > 0"
    assert speed_dist["average_path_distance_km"] > 0, "Expected average path distance > 0"

    # 5. Common Vehicle Routes
    print("\n--- 5. Most Common Multi-Camera Vehicle Routes ---")
    routes = engine.get_most_common_vehicle_routes()
    if routes:
        for idx, r in enumerate(routes, 1):
            print(f"  Route #{idx}: {r['route']}")
            print(f"    - Frequency: {r['frequency']} vehicle(s) | Vehicles: {', '.join(r['vehicles'])}")
    else:
        print("  No multi-camera routes found.")

    # 6. JSON Export
    json_path = "data/traffic_analytics.json"
    print(f"\n--- 6. Exporting Analytics Payload to JSON ---")
    out_file = engine.export_analytics_json(json_path)
    print(f"  [PASS] Analytics payload exported to: {out_file}")
    assert os.path.exists(out_file), "JSON export file not found"
    
    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "summary" in loaded
        assert "cameras" in loaded
        assert "hourly_traffic" in loaded
        assert "speed_and_distance" in loaded
        print(f"  [PASS] JSON structure validated ({os.path.getsize(out_file)} bytes).")

    # 7. Safety / Non-regression Check
    # Confirm 52 records preserved in PostgreSQL and SQLite
    import sqlite3
    sq = sqlite3.connect("file:data/vehicle_history.db?mode=ro", uri=True)
    sq_cnt = sq.cursor().execute("SELECT COUNT(*) FROM vehicle_detections").fetchone()[0]
    sq.close()
    assert sq_cnt == 52, "SQLite record count modified!"

    print("\n" + "=" * 75)
    print("TraceX STEP 15 Traffic Analytics Test Summary: ALL PASS")
    print("=" * 75)
    return True

if __name__ == "__main__":
    success = test_traffic_analytics()
    sys.exit(0 if success else 1)
