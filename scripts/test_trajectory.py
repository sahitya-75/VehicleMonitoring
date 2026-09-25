"""
TraceX SIH 2026 - Trajectory Reconstruction Verification Script
Tests exact plate trajectory reconstruction for 'HR51BC3493' and fuzzy matching for 'HR51BC3498'.
"""

import sys
import os
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trajectory.trajectory_service import TrajectoryService, levenshtein_distance

def test_trajectory_engine():
    print("=" * 75)
    print("TraceX STEP 13 - Trajectory Reconstruction Test Suite")
    print("=" * 75)

    service = TrajectoryService()

    # 1. Test Levenshtein distance function
    print("\n--- Test 1: Levenshtein Distance Algorithm ---")
    pairs = [
        ("HR51BC3493", "HR51BC3493", 0),
        ("HR51BC3493", "HR51BC3498", 1),  # 3 -> 8
        ("HR51BC3493", "HR51BC3490", 1),  # 3 -> 0
        ("DL3CC8387", "DL3CC8388", 1),
    ]
    lev_pass = True
    for s1, s2, expected in pairs:
        actual = levenshtein_distance(s1, s2)
        match = (actual == expected)
        print(f"  Levenshtein('{s1}', '{s2}') = {actual} (Expected: {expected}) -> {'PASS' if match else 'FAIL'}")
        if not match:
            lev_pass = False

    # 2. Test Exact Match for Known Plate 'HR51BC3493'
    test_plate_exact = "HR51BC3493"
    print(f"\n--- Test 2: Exact Trajectory Query for '{test_plate_exact}' ---")
    exact_traj = service.get_exact_trajectory(test_plate_exact)

    if exact_traj is None:
        print(f"[FAIL] No trajectory found for exact plate '{test_plate_exact}'")
        exact_pass = False
    else:
        exact_pass = True
        print(f"[PASS] Exact Trajectory Found:")
        print(f"  Matched Plate        : {exact_traj.matched_plate}")
        print(f"  Match Type           : {exact_traj.match_type} (Edit Distance: {exact_traj.edit_distance})")
        print(f"  Total Sightings      : {exact_traj.total_sightings}")
        print(f"  First Seen           : {exact_traj.first_seen}")
        print(f"  Last Seen            : {exact_traj.last_seen}")
        print(f"  Total Distance (m)   : {exact_traj.total_distance_meters:.2f} m ({exact_traj.total_distance_meters/1000:.2f} km)")
        print(f"  Unique Cameras ({len(exact_traj.unique_cameras)}) : {', '.join(exact_traj.unique_cameras)}")

        print("\n  Chronological Sightings / Trajectory Waypoints:")
        print(f"  {'#':<3} {'Camera ID':<25} {'Timestamp':<28} {'Coordinates (Lat, Lon)':<25} {'Dist (m)':<10} {'Speed (km/h)':<12}")
        print("  " + "-" * 105)
        for idx, pt in enumerate(exact_traj.points, 1):
            coords_str = f"({pt.latitude:.4f}, {pt.longitude:.4f})"
            print(f"  {idx:<3} {pt.camera_id:<25} {pt.timestamp:<28} {coords_str:<25} {pt.distance_from_prev_m:<10.1f} {pt.speed_kmh:<12.1f}")

    # 3. Test Fuzzy Match with OCR Variation 'HR51BC3498' (3 replaced by 8)
    test_plate_fuzzy = "HR51BC3498"
    print(f"\n--- Test 3: Fuzzy Trajectory Query for OCR Variation '{test_plate_fuzzy}' ---")
    fuzzy_results = service.query_trajectory(test_plate_fuzzy, allow_fuzzy=True, max_edit_distance=2)

    if not fuzzy_results:
        print(f"[FAIL] No fuzzy matches found for '{test_plate_fuzzy}' with max_edit_distance=2")
        fuzzy_pass = False
    else:
        matched_plates = [t.matched_plate for t in fuzzy_results]
        print(f"[PASS] Found {len(fuzzy_results)} candidate vehicle trajectory match(es): {matched_plates}")
        fuzzy_pass = (test_plate_exact in matched_plates)
        for t in fuzzy_results:
            print(f"  * Candidate: {t.matched_plate:<15} | Match: {t.match_type} (Dist: {t.edit_distance}) | Sightings: {t.total_sightings} | Dist: {t.total_distance_meters/1000:.2f} km")

    # 4. Verification Summary
    print("\n" + "=" * 75)
    print("TraceX STEP 13 Trajectory Test Summary")
    print("=" * 75)
    print(f"  Levenshtein Metric Test   : {'PASS' if lev_pass else 'FAIL'}")
    print(f"  Exact Trajectory Query    : {'PASS' if exact_pass else 'FAIL'}")
    print(f"  Fuzzy Trajectory Query    : {'PASS' if fuzzy_pass else 'FAIL'}")
    print(f"  HR51BC3493 Sightings Count: {exact_traj.total_sightings if exact_traj else 0}")
    print("=" * 75)

    return lev_pass and exact_pass and fuzzy_pass

if __name__ == "__main__":
    success = test_trajectory_engine()
    sys.exit(0 if success else 1)
