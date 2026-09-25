"""
TraceX SIH 2026 - Complete Dashboard & Frontend Integration Test Suite
Verifies frontend assets, static servers, API integration, trajectory data, and watchlist flows.
"""

import sys
import os
import json
import urllib.request
import urllib.error

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FRONTEND_URL = "http://127.0.0.1:3000"
BACKEND_URL = "http://127.0.0.1:8000"


def fetch_url(url: str, method: str = "GET", data: dict = None):
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            content = response.read().decode("utf-8")
            headers_dict = dict(response.getheaders())
            return status, content, headers_dict
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        return e.code, content, {}
    except Exception as e:
        return 0, str(e), {}


def test_dashboard():
    print("=" * 75)
    print("TraceX STEP 19 - React Dashboard & Frontend Verification Suite")
    print("=" * 75)

    results = {}

    # 1. Frontend Root Delivery
    print("\n--- 1. Testing Frontend Static Server (http://localhost:3000) ---")
    st, content, hdrs = fetch_url(f"{FRONTEND_URL}/")
    if st == 200 and "TraceX" in content and "id=\"root\"" in content:
        results["Frontend Index Delivery"] = "PASS"
        print(f"  [PASS] GET / -> HTTP 200 | Content Length: {len(content)} bytes | React Root Present")
    else:
        results["Frontend Index Delivery"] = f"FAIL (HTTP {st})"
        print(f"  [FAIL] {st}: {content[:100]}")

    # 2. Frontend Assets (App.jsx, styles.css, api.js)
    print("\n--- 2. Testing Frontend Static Assets ---")
    st_app, content_app, hdrs_app = fetch_url(f"{FRONTEND_URL}/src/App.jsx")
    st_css, content_css, hdrs_css = fetch_url(f"{FRONTEND_URL}/styles.css")
    st_api, content_api, hdrs_api = fetch_url(f"{FRONTEND_URL}/src/api.js")

    if st_app == 200 and st_css == 200 and st_api == 200:
        results["Frontend Assets Delivery"] = "PASS"
        print(f"  [PASS] /src/App.jsx  : HTTP {st_app} ({len(content_app)} bytes)")
        print(f"  [PASS] /styles.css   : HTTP {st_css} ({len(content_css)} bytes)")
        print(f"  [PASS] /src/api.js   : HTTP {st_api} ({len(content_api)} bytes)")
    else:
        results["Frontend Assets Delivery"] = "FAIL"
        print(f"  [FAIL] Asset statuses: App.jsx={st_app}, styles.css={st_css}, api.js={st_api}")

    # 3. FastAPI Dashboard Mount (/dashboard)
    print("\n--- 3. Testing FastAPI Static Mount (http://localhost:8000/dashboard/) ---")
    st_mount, content_mount, _ = fetch_url(f"{BACKEND_URL}/dashboard/")
    if st_mount == 200 and "TraceX" in content_mount:
        results["FastAPI Dashboard Mount"] = "PASS"
        print(f"  [PASS] GET /dashboard/ -> HTTP 200 | Dashboard accessible via API gateway")
    else:
        results["FastAPI Dashboard Mount"] = f"FAIL (HTTP {st_mount})"
        print(f"  [FAIL] Mount status: {st_mount}")

    # 4. Backend Health Check
    print("\n--- 4. Testing Backend API Health ---")
    st_h, content_h, _ = fetch_url(f"{BACKEND_URL}/api/health")
    if st_h == 200:
        h_data = json.loads(content_h)
        if h_data.get("status") == "HEALTHY":
            results["Backend API Health"] = "PASS"
            print(f"  [PASS] API Status: HEALTHY | Engine: {h_data['database']['engine']} | PostGIS: {h_data['database']['postgis_version']}")
        else:
            results["Backend API Health"] = f"FAIL (Status: {h_data.get('status')})"
    else:
        results["Backend API Health"] = f"FAIL (HTTP {st_h})"

    # 5. Vehicle Trajectory Load for Known Plate 'HR51BC3493'
    print("\n--- 5. Testing HR51BC3493 Trajectory & Map Dataset ---")
    st_traj, content_traj, _ = fetch_url(f"{BACKEND_URL}/api/vehicles/HR51BC3493/trajectory")
    if st_traj == 200:
        traj = json.loads(content_traj)
        pts = traj.get("points", [])
        dist_km = traj.get("total_distance_meters", 0) / 1000.0
        cams = traj.get("unique_cameras", [])

        if len(pts) == 4 and round(dist_km, 2) == 39.66 and len(cams) == 4:
            results["HR51BC3493 Trajectory Display"] = "PASS"
            print(f"  [PASS] Target Plate    : {traj['matched_plate']}")
            print(f"  [PASS] Waypoint Nodes  : {len(pts)} Surveillance Stations")
            print(f"  [PASS] Total Distance  : {dist_km:.2f} km")
            print(f"  [PASS] Cameras Visited : {', '.join(cams)}")
        else:
            results["HR51BC3493 Trajectory Display"] = f"FAIL (Nodes: {len(pts)}, Dist: {dist_km} km)"
    else:
        results["HR51BC3493 Trajectory Display"] = f"FAIL (HTTP {st_traj})"

    # 6. Fuzzy Vehicle Search
    print("\n--- 6. Testing Fuzzy Search (HR51BC3498) ---")
    st_fuzz, content_fuzz, _ = fetch_url(f"{BACKEND_URL}/api/vehicles/search?plate=HR51BC3498&fuzzy=true")
    if st_fuzz == 200:
        fuzz_data = json.loads(content_fuzz)
        matches = [r["matched_plate"] for r in fuzz_data.get("results", [])]
        if "HR51BC3493" in matches:
            results["Fuzzy Search Engine"] = "PASS"
            print(f"  [PASS] Query 'HR51BC3498' -> Successfully matched '{matches[0]}' (Edit Distance: 1)")
        else:
            results["Fuzzy Search Engine"] = f"FAIL (Matches: {matches})"
    else:
        results["Fuzzy Search Engine"] = f"FAIL (HTTP {st_fuzz})"

    # 7. Analytics Data Feed
    print("\n--- 7. Testing Analytics Payload Delivery ---")
    st_ana, content_ana, _ = fetch_url(f"{BACKEND_URL}/api/analytics/summary")
    if st_ana == 200:
        ana = json.loads(content_ana)
        if "summary" in ana and "cameras" in ana and "speed_and_distance" in ana:
            results["Traffic Analytics Feed"] = "PASS"
            print(f"  [PASS] Analytics Payload: {len(ana['cameras'])} cameras, {ana['summary']['total_detections']} detections, avg speed {ana['speed_and_distance']['average_speed_kmh']} km/h")
        else:
            results["Traffic Analytics Feed"] = "FAIL (Missing sections)"
    else:
        results["Traffic Analytics Feed"] = f"FAIL (HTTP {st_ana})"

    # 8. Alerts & Watchlist Delivery
    print("\n--- 8. Testing Alerts & Watchlist Payload Delivery ---")
    st_alt, content_alt, _ = fetch_url(f"{BACKEND_URL}/api/alerts")
    st_wat, content_wat, _ = fetch_url(f"{BACKEND_URL}/api/watchlist")

    if st_alt == 200 and st_wat == 200:
        alts = json.loads(content_alt)
        wats = json.loads(content_wat)
        results["Alerts & Watchlist Feed"] = "PASS"
        print(f"  [PASS] Active Alerts Loaded    : {len(alts)} alert items")
        print(f"  [PASS] Active Watchlist Loaded : {len(wats)} watchlist entries")
    else:
        results["Alerts & Watchlist Feed"] = f"FAIL (Alerts={st_alt}, Watchlist={st_wat})"

    # Final Summary Table
    print("\n" + "=" * 75)
    print("TraceX STEP 19 React Dashboard Verification Summary")
    print("=" * 75)
    all_pass = True
    for item, status in results.items():
        is_p = status.startswith("PASS")
        print(f"  {item:<32}: {status}")
        if not is_p:
            all_pass = False

    return all_pass

if __name__ == "__main__":
    success = test_dashboard()
    sys.exit(0 if success else 1)
