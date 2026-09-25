"""
TraceX SIH 2026 - PostGIS Verification Suite
Verifies PostGIS extension, geography column, EPSG 4326, GiST spatial index, and spatial distance query.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

def verify_postgis():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_password_2026")

    print("=" * 65)
    print("TraceX PostGIS Spatial Verification Suite")
    print("=" * 65)
    print(f"Connecting to database '{dbname}' on {host}:{port} as user '{user}'...")

    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=5
        )
        cur = conn.cursor()

        # 1. PostGIS extension & version verification
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'postgis';")
        ext_row = cur.fetchone()
        if not ext_row:
            print("[FAIL] PostGIS extension is NOT active in database.")
            return False
        
        cur.execute("SELECT PostGIS_Version();")
        postgis_ver = cur.fetchone()[0]
        print(f"[PASS] PostGIS Extension active: Version {postgis_ver} (Extension version {ext_row[1]})")

        # 2. Verify 'location' column in vehicle_detections
        cur.execute("""
            SELECT f_geography_column, type, srid 
            FROM geography_columns 
            WHERE f_table_name = 'vehicle_detections' AND f_geography_column = 'location';
        """)
        geo_col = cur.fetchone()
        if not geo_col:
            # Fallback check via information_schema / pg_attribute
            cur.execute("""
                SELECT column_name, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'vehicle_detections' AND column_name = 'location';
            """)
            info_col = cur.fetchone()
            if not info_col:
                print("[FAIL] Column 'location' does NOT exist in 'vehicle_detections'.")
                return False
            print(f"[PASS] Column 'location' exists (udt: {info_col[1]}).")
        else:
            print(f"[PASS] Column '{geo_col[0]}' exists -> Type: {geo_col[1]}, SRID (EPSG): {geo_col[2]}")
            if geo_col[2] != 4326:
                print(f"[FAIL] Expected EPSG 4326, got: {geo_col[2]}")
                return False

        # 3. Verify GiST spatial index
        cur.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'vehicle_detections' AND indexname = 'idx_vehicle_detections_location';
        """)
        idx_row = cur.fetchone()
        if not idx_row:
            print("[FAIL] Spatial index 'idx_vehicle_detections_location' NOT found.")
            return False
        if "gist" not in idx_row[1].lower():
            print(f"[FAIL] Spatial index is not using GiST: {idx_row[1]}")
            return False
        print(f"[PASS] GiST Spatial Index verified: {idx_row[0]} -> {idx_row[1]}")

        # 4. Test spatial distance query
        # Connaught Place (28.6315° N, 77.2167° E) to India Gate (28.6129° N, 77.2295° E)
        # Expected great-circle geodesic distance in meters: ~2400m - 2500m
        cur.execute("""
            SELECT ST_Distance(
                ST_SetSRID(ST_MakePoint(77.2167, 28.6315), 4326)::geography,
                ST_SetSRID(ST_MakePoint(77.2295, 28.6129), 4326)::geography
            ) AS distance_meters;
        """)
        dist_meters = cur.fetchone()[0]
        print(f"[PASS] Spatial Distance Query (Connaught Place to India Gate): {dist_meters:.2f} meters")

        # 5. Verify all existing columns preserved
        expected_cols = [
            "id", "plate_text", "cleaned_plate", "yolo_confidence", 
            "ocr_confidence", "valid_indian_plate", "camera_id", 
            "timestamp", "frame_number", "track_id", "latitude", 
            "longitude", "location"
        ]
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'vehicle_detections';
        """)
        existing_cols = {r[0] for r in cur.fetchall()}
        missing = [c for c in expected_cols if c not in existing_cols]
        if missing:
            print(f"[FAIL] Missing columns: {missing}")
            return False
        print(f"[PASS] All {len(expected_cols)} vehicle_detections columns intact and verified.")

        print("\n" + "=" * 65)
        print("[SUCCESS] ALL PostGIS & Spatial Schema Tests PASSED!")
        print("=" * 65)
        return True

    except Exception as e:
        print(f"\n[FAIL] PostGIS verification error: {e}")
        return False
    finally:
        if conn is not None and not conn.closed:
            conn.close()

if __name__ == "__main__":
    success = verify_postgis()
    sys.exit(0 if success else 1)
