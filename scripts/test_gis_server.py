"""
TraceX SIH 2026 - GIS Server Verification Test
"""

import urllib.request
import json

def test_gis_server():
    base_url = "http://127.0.0.1:8080"
    
    print("Testing GIS Server Endpoints:")
    
    # 1. Test GET /
    with urllib.request.urlopen(f"{base_url}/") as res:
        status = res.status
        html = res.read().decode("utf-8")
        assert status == 200
        assert "TraceX" in html
        print(f"  [PASS] GET / -> HTTP {status}, {len(html)} bytes")

    # 2. Test GET /api/trajectory?plate=HR51BC3493
    with urllib.request.urlopen(f"{base_url}/api/trajectory?plate=HR51BC3493") as res:
        data = json.loads(res.read().decode("utf-8"))
        assert data["count"] == 1
        traj = data["trajectories"][0]
        assert traj["total_sightings"] == 4
        assert len(traj["unique_cameras"]) == 4
        print(f"  [PASS] GET /api/trajectory?plate=HR51BC3493 -> {traj['total_sightings']} sightings, {traj['total_distance_meters']/1000:.2f} km")

    # 3. Test GET /api/trajectory?plate=HR51BC3498&fuzzy=true
    with urllib.request.urlopen(f"{base_url}/api/trajectory?plate=HR51BC3498&fuzzy=true") as res:
        data = json.loads(res.read().decode("utf-8"))
        assert data["count"] >= 1
        cand = data["trajectories"][0]
        assert cand["matched_plate"] == "HR51BC3493"
        print(f"  [PASS] GET /api/trajectory?plate=HR51BC3498 [Fuzzy] -> Matched {cand['matched_plate']} (edit dist {cand['edit_distance']})")

    # 4. Test GET /api/plates
    with urllib.request.urlopen(f"{base_url}/api/plates") as res:
        data = json.loads(res.read().decode("utf-8"))
        assert data["count"] > 0
        print(f"  [PASS] GET /api/plates -> {data['count']} unique plates indexed")

    print("\n[SUCCESS] All GIS Web Server Endpoints are active and verified!")
    return True

if __name__ == "__main__":
    test_gis_server()
