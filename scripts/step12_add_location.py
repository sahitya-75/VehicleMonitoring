"""
TraceX SIH 2026 - STEP 12.4: Add PostGIS Geographic Column
Adds location geography(POINT, 4326) to vehicle_detections and populates it from latitude/longitude.
Does NOT create spatial indexes yet.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

def add_location_column():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    superuser = os.getenv("PG_SUPERUSER", "postgres")
    superpass = os.getenv("PG_SUPERPASSWORD")
    app_user = os.getenv("DB_USER", "tracex_user")

    print("=" * 65)
    print("TraceX STEP 12.4 - Add PostGIS Geographic Column")
    print("=" * 65)
    print(f"Connecting to database '{dbname}' on {host}:{port} as superuser '{superuser}'...")

    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=superuser,
            password=superpass,
            connect_timeout=5
        )
        conn.autocommit = True
        cur = conn.cursor()

        # 0. Check initial row counts
        cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        initial_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM vehicle_detections WHERE latitude IS NOT NULL AND longitude IS NOT NULL;")
        coords_count = cur.fetchone()[0]
        print(f"Initial total rows in vehicle_detections: {initial_count}")
        print(f"Rows with non-null (latitude, longitude): {coords_count}")

        # 1. Add column if not exists
        print("\nAdding column: location geography(POINT, 4326)...")
        cur.execute("""
            ALTER TABLE vehicle_detections 
            ADD COLUMN IF NOT EXISTS location geography(POINT, 4326);
        """)
        print("[OK] Column 'location geography(POINT, 4326)' added / verified.")

        # 2. Populate location column
        print("Populating 'location' from latitude and longitude...")
        cur.execute("""
            UPDATE vehicle_detections 
            SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL 
              AND location IS NULL;
        """)
        rows_updated = cur.rowcount
        print(f"[OK] Rows updated: {rows_updated}")

        # 3. Grant privileges on table to tracex_user
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON TABLE vehicle_detections TO {};").format(sql.Identifier(app_user)))
        print(f"[OK] Privileges granted to '{app_user}'.")

        # 4. Verification checks
        print("\n" + "=" * 30 + " Verification " + "=" * 30)

        # Verify location column in geography_columns
        cur.execute("""
            SELECT f_table_name, f_geography_column, type, srid 
            FROM geography_columns 
            WHERE f_table_name = 'vehicle_detections' AND f_geography_column = 'location';
        """)
        geo_info = cur.fetchone()
        if geo_info:
            print(f"[PASS] Geography Column: {geo_info[1]}, Type: {geo_info[2]}, SRID: {geo_info[3]}")
        else:
            # Fallback information_schema check
            cur.execute("""
                SELECT column_name, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'vehicle_detections' AND column_name = 'location';
            """)
            col_info = cur.fetchone()
            print(f"[PASS] Column info: {col_info[0]} -> {col_info[1]}")

        # Verify row counts
        cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        final_total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM vehicle_detections WHERE latitude IS NOT NULL AND longitude IS NOT NULL;")
        final_coords_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM vehicle_detections WHERE location IS NOT NULL;")
        final_location_count = cur.fetchone()[0]

        print(f"[PASS] Total records in vehicle_detections: {final_total} (Unchanged: {final_total == initial_count})")
        print(f"[PASS] Records with non-null latitude/longitude: {final_coords_count}")
        print(f"[PASS] Records with non-null location: {final_location_count}")

        print("=" * 65)
        return True

    except Exception as e:
        print(f"\n[FAIL] Error during STEP 12.4:\n{e}")
        print("=" * 65)
        return False
    finally:
        if conn is not None and not conn.closed:
            conn.close()

if __name__ == "__main__":
    success = add_location_column()
    sys.exit(0 if success else 1)
