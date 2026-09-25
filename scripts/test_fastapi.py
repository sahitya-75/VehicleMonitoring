"""
TraceX SIH 2026 - FastAPI Backend Test Suite
Tests every REST endpoint against the running FastAPI application.
"""

import sys
import os
import json
import urllib.request
import urllib.error

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_URL = "http://127.0.0.1:8000"


def make_request(path: str, method: str = "GET", data: dict = None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            content = response.read().decode("utf-8")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = content
            return status, parsed
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = content
        return e.code, parsed
    except Exception as e:
        return 0, str(e)


def run_fastapi_tests():
    print("=" * 75)
    print("TraceX STEP 18 - FastAPI Backend Verification Suite")
    print("=" * 75)

    results = {}

    # 1. Test /api/health
    print("\n--- 1. Testing GET /api/health ---")
    status, res = make_request("/api/health")
    if status == 200 and res.get("status") == "HEALTHY":
        results["GET /api/health"] = "PASS"
        print(f"  [PASS] Status: {status} | API: {res.get('status')} | DB: {res.get('database', {}).get('status')}")
        print(f"         PostgreSQL: {res.get('database', {}).get('version')[:40]}...")
        print(f"         PostGIS   : {res.get('database', {}).get('postgis_version')}")
    else:
        results["GET /api/health"] = f"FAIL (HTTP {status}: {res})"
        print(f"  [FAIL] {res}")

    # 2. Test GET /api/vehicles (Pagination & Filter)
    print("\n--- 2. Testing GET /api/vehicles ---")
    status, res = make_request("/api/vehicles?page=1&limit=5")
    if status == 200 and "records" in res and res.get("total") == 52:
        results["GET /api/vehicles"] = "PASS"
        print(f"  [PASS] Status: {status} | Total Detections: {res['total']} | Page Records: {len(res['records'])}")
    else:
        results["GET /api/vehicles"] = f"FAIL (HTTP {status}: {res})"
        print(f"  [FAIL] {res}")

    # 3. Test GET /api/vehicles/search?plate=HR51BC3493
    print("\n--- 3. Testing GET /api/vehicles/search (Fuzzy Match) ---")
    status, res = make_request("/api/vehicles/search?plate=HR51BC3498&fuzzy=true")
    if status == 200 and res.get("count", 0) >= 1:
        matched = res["results"][0]["matched_plate"]
        results["GET /api/vehicles/search"] = f"PASS (Matched: {matched})"
        print(f"  [PASS] Status: {status} | Query: HR51BC3498 -> Matched: {matched} (Dist: {res['results'][0]['edit_distance']})")
    else:
        results["GET /api/vehicles/search"] = f"FAIL (HTTP {status}: {res})"
        print(f"  [FAIL] {res}")

    # 4. Test GET /api/vehicles/{plate}
    print("\n--- 4. Testing GET /api/vehicles/HR51BC3493 ---")
    status, res = make_request("/api/vehicles/HR51BC3493")
    if status == 200 and res.get("plate") == "HR51BC3493" and res.get("total_sightings") == 4:
        results["GET /api/vehicles/{plate}"] = "PASS"
        print(f"  [PASS] Status: {status} | Sightings: {res['total_sightings']} | Distance: {res['total_distance_km']} km")
    else:
        results["GET /api/vehicles/{plate}"] = f"FAIL (HTTP {status}: {res})"
        print(f"  [FAIL] {res}")

    # 5. Test GET /api/vehicles/{plate}/trajectory
    print("\n--- 5. Testing GET /api/vehicles/HR51BC3493/trajectory ---")
    status, res = make_request("/api/vehicles/HR51BC3493/trajectory")
    if status == 200 and len(res.get("points", [])) == 4 and res.get("total_distance_meters", 0) > 30000:
        results["GET /api/vehicles/{plate}/trajectory"] = "PASS"
        print(f"  [PASS] Status: {status} | Path Nodes: {len(res['points'])} | Dist: {res['total_distance_meters']:.1f} m")
    else:
        results["GET /api/vehicles/{plate}/trajectory"] = f"FAIL (HTTP {status}: {res})"
        print(f"  [FAIL] {res}")

    # 6. Test 404 on Non-existent plate
    print("\n--- 6. Testing 404 on Non-existent Plate ---")
    status, res = make_request("/api/vehicles/INVALID9999")
    if status == 404:
        results["404 Validation"] = "PASS"
        print(f"  [PASS] Correct 404 returned for invalid plate: {res.get('detail')}")
    else:
        results["404 Validation"] = f"FAIL (Expected 404, got {status})"
        print(f"  [FAIL] {res}")

    # 7. Test Analytics Endpoints
    print("\n--- 7. Testing Analytics Endpoints ---")
    status1, res1 = make_request("/api/analytics/summary")
    status2, res2 = make_request("/api/analytics/cameras")
    status3, res3 = make_request("/api/analytics/hourly")

    if status1 == 200 and status2 == 200 and status3 == 200:
        results["GET /api/analytics/summary"] = "PASS"
        results["GET /api/analytics/cameras"] = "PASS"
        results["GET /api/analytics/hourly"] = "PASS"
        print(f"  [PASS] /api/analytics/summary: Status 200 ({len(res1)} sections)")
        print(f"  [PASS] /api/analytics/cameras: Status 200 ({len(res2)} cameras)")
        print(f"  [PASS] /api/analytics/hourly : Status 200 ({len(res3)} time slots)")
    else:
        results["GET /api/analytics/*"] = "FAIL"
        print(f"  [FAIL] Statuses: {status1}, {status2}, {status3}")

    # 8. Test Alerts & Watchlist Endpoints
    print("\n--- 8. Testing Alerts & Watchlist Endpoints ---")
    # GET /api/alerts
    status_a, res_a = make_request("/api/alerts")
    if status_a == 200 and len(res_a) > 0:
        results["GET /api/alerts"] = f"PASS ({len(res_a)} alerts)"
        print(f"  [PASS] GET /api/alerts: Status 200 ({len(res_a)} alerts loaded)")
    else:
        results["GET /api/alerts"] = f"FAIL (HTTP {status_a}: {res_a})"

    # GET /api/watchlist
    status_w, res_w = make_request("/api/watchlist")
    if status_w == 200:
        results["GET /api/watchlist"] = f"PASS ({len(res_w)} items)"
        print(f"  [PASS] GET /api/watchlist: Status 200 ({len(res_w)} items)")
    else:
        results["GET /api/watchlist"] = f"FAIL (HTTP {status_w}: {res_w})"

    # POST /api/watchlist (Add Temporary Test Vehicle)
    test_watch_payload = {
        "cleaned_plate": "TEST9999",
        "description": "Integration Test Watchlist Entry",
        "priority": "HIGH",
        "enabled": True
    }
    status_post, res_post = make_request("/api/watchlist", method="POST", data=test_watch_payload)
    if status_post == 201 and res_post.get("cleaned_plate") == "TEST9999":
        results["POST /api/watchlist"] = "PASS"
        print(f"  [PASS] POST /api/watchlist: Status 201 (Added {res_post['cleaned_plate']})")
    else:
        results["POST /api/watchlist"] = f"FAIL (HTTP {status_post}: {res_post})"

    # DELETE /api/watchlist/TEST9999
    status_del, res_del = make_request("/api/watchlist/TEST9999", method="DELETE")
    if status_del == 200 and res_del.get("success") is True:
        results["DELETE /api/watchlist/{plate}"] = "PASS"
        print(f"  [PASS] DELETE /api/watchlist/TEST9999: Status 200 ({res_del['message']})")
    else:
        results["DELETE /api/watchlist/{plate}"] = f"FAIL (HTTP {status_del}: {res_del})"

    # Final Summary
    print("\n" + "=" * 75)
    print("TraceX STEP 18 FastAPI Endpoint Test Summary")
    print("=" * 75)
    all_pass = True
    for ep, stat in results.items():
        is_p = stat.startswith("PASS")
        print(f"  {ep:<35}: {stat}")
        if not is_p:
            all_pass = False

    return all_pass

if __name__ == "__main__":
    success = run_fastapi_tests()
    sys.exit(0 if success else 1)
