"""
TraceX SIH 2026 - STEP 12.5: Spatial Index & PostGIS Complete Verification Suite
Creates GiST index on location and executes comprehensive spatial and schema verification.
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv
import psycopg2

def run_step_12_5():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_password_2026")
    superuser = os.getenv("PG_SUPERUSER", "postgres")
    superpass = os.getenv("PG_SUPERPASSWORD")

    results = {}

    print("=" * 70)
    print("TraceX STEP 12.5 - Spatial Index & PostGIS Verification")
    print("=" * 70)

    # 1. Create GiST spatial index (using superuser or owner)
    print("1. Creating GiST spatial index 'idx_vehicle_detections_location'...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=superuser, password=superpass, connect_timeout=5
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_vehicle_detections_location 
                ON vehicle_detections USING GIST (location);
            """)
        conn.close()
        results["CREATE_GIST_INDEX"] = "PASS"
        print("   [OK] Spatial GiST index creation command executed successfully.")
    except Exception as e:
        results["CREATE_GIST_INDEX"] = f"FAIL: {e}"
        print(f"   [FAIL] {e}")

    # 2. Connect as tracex_user to test standard application access
    print("\n2. Connecting as application user 'tracex_user' to verify...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=5
        )
        cur = conn.cursor()

        # Check PostGIS_Version()
        cur.execute("SELECT PostGIS_Version();")
        pg_ver = cur.fetchone()[0]
        results["POSTGIS_VERSION"] = f"PASS ({pg_ver})"
        print(f"   - PostGIS_Version(): PASS -> {pg_ver}")

        # Check location column data type & SRID in geography_columns
        cur.execute("""
            SELECT f_geography_column, type, srid 
            FROM geography_columns 
            WHERE f_table_name = 'vehicle_detections' AND f_geography_column = 'location';
        """)
        geo_info = cur.fetchone()
        if geo_info and geo_info[1].upper() == "POINT" and geo_info[2] == 4326:
            results["LOCATION_DATA_TYPE"] = f"PASS ({geo_info[1]})"
            results["LOCATION_SRID"] = f"PASS ({geo_info[2]})"
            print(f"   - location column data type: PASS -> {geo_info[1]}")
            print(f"   - location SRID: PASS -> {geo_info[2]}")
        else:
            results["LOCATION_DATA_TYPE"] = "FAIL"
            results["LOCATION_SRID"] = "FAIL"
            print(f"   - location column check: FAIL -> {geo_info}")

        # Check GiST spatial index exists and uses gist method
        cur.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'vehicle_detections' AND indexname = 'idx_vehicle_detections_location';
        """)
        gist_idx = cur.fetchone()
        if gist_idx and "gist" in gist_idx[1].lower():
            results["GIST_SPATIAL_INDEX"] = "PASS"
            print(f"   - GiST spatial index exists: PASS -> {gist_idx[0]}")
        else:
            results["GIST_SPATIAL_INDEX"] = "FAIL"
            print(f"   - GiST spatial index exists: FAIL -> {gist_idx}")

        # Check all 4 existing B-tree indexes
        required_btree_indexes = [
            "idx_vehicle_detections_cleaned_plate",
            "idx_vehicle_detections_timestamp",
            "idx_vehicle_detections_camera_id",
            "idx_vehicle_detections_track_id"
        ]
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'vehicle_detections';
        """)
        present_indexes = {row[0] for row in cur.fetchall()}
        missing_btree = [idx for idx in required_btree_indexes if idx not in present_indexes]
        if not missing_btree:
            results["4_BTREE_INDEXES"] = "PASS (all 4 present)"
            print("   - Existing 4 B-tree indexes exist: PASS")
            for idx in required_btree_indexes:
                print(f"       * {idx}: PASS")
        else:
            results["4_BTREE_INDEXES"] = f"FAIL (missing {missing_btree})"
            print(f"   - Existing 4 B-tree indexes: FAIL (missing {missing_btree})")

        # Check PostgreSQL record count
        cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        pg_count = cur.fetchone()[0]
        if pg_count == 0:
            results["PG_RECORD_COUNT"] = "PASS (0 records)"
            print(f"   - PostgreSQL vehicle_detections count: PASS -> {pg_count} records")
        else:
            results["PG_RECORD_COUNT"] = f"FAIL (unexpected count: {pg_count})"
            print(f"   - PostgreSQL vehicle_detections count: FAIL -> {pg_count}")

        # 3. Test Spatial query functionality (ST_MakePoint, ST_SetSRID, ST_Distance, ST_DWithin)
        cur.execute("""
            WITH test_points AS (
                SELECT 
                    ST_SetSRID(ST_MakePoint(77.2167, 28.6315), 4326)::geography AS p1_connaught_place,
                    ST_SetSRID(ST_MakePoint(77.2295, 28.6129), 4326)::geography AS p2_india_gate
            )
            SELECT 
                ST_Distance(p1_connaught_place, p2_india_gate) AS distance_meters,
                ST_DWithin(p1_connaught_place, p2_india_gate, 3000) AS within_3km,
                ST_DWithin(p1_connaught_place, p2_india_gate, 1000) AS within_1km
            FROM test_points;
        """)
        spatial_row = cur.fetchone()
        dist_m, within_3k, within_1k = spatial_row
        if 2000 < dist_m < 3000 and within_3k is True and within_1k is False:
            results["SPATIAL_FUNCTIONS_TEST"] = f"PASS (ST_Distance={dist_m:.1f}m, ST_DWithin(3km)={within_3k}, ST_DWithin(1km)={within_1k})"
            print(f"   - Spatial query execution: PASS -> ST_Distance={dist_m:.2f}m, ST_DWithin 3km={within_3k}, 1km={within_1k}")
        else:
            results["SPATIAL_FUNCTIONS_TEST"] = f"FAIL (unexpected output: {spatial_row})"
            print(f"   - Spatial query execution: FAIL -> {spatial_row}")

        conn.close()
    except Exception as e:
        print(f"   [FAIL] Error during PostgreSQL checks: {e}")
        results["PG_CHECKS"] = f"FAIL: {e}"

    # 4. Check SQLite database backup
    sqlite_path = os.path.join("data", "vehicle_history.db")
    print(f"\n3. Checking SQLite database integrity at '{sqlite_path}'...")
    try:
        sqlite_conn = sqlite3.connect(sqlite_path)
        sq_cur = sqlite_conn.cursor()
        sq_cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        sqlite_count = sq_cur.fetchone()[0]
        sqlite_conn.close()
        if sqlite_count == 52:
            results["SQLITE_BACKUP_INTEGRITY"] = f"PASS ({sqlite_count} records intact)"
            print(f"   - SQLite backup record count: PASS -> {sqlite_count} records untouched")
        else:
            results["SQLITE_BACKUP_INTEGRITY"] = f"FAIL (count={sqlite_count}, expected 52)"
            print(f"   - SQLite backup: FAIL (count={sqlite_count})")
    except Exception as e:
        results["SQLITE_BACKUP_INTEGRITY"] = f"FAIL: {e}"
        print(f"   - SQLite backup check error: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("STEP 12.5 Verification Summary")
    print("=" * 70)
    all_pass = True
    for test_name, status in results.items():
        is_pass = status.startswith("PASS")
        print(f"  {test_name:<30}: {status}")
        if not is_pass:
            all_pass = False

    return all_pass

if __name__ == "__main__":
    success = run_step_12_5()
    sys.exit(0 if success else 1)
