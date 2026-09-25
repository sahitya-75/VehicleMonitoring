"""
TraceX SIH 2026 - Enable PostGIS & Spatial Schema Setup
Enables PostGIS extension in 'tracex' database, adds geography column, and creates GiST spatial index.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

def enable_postgis():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    superuser = os.getenv("PG_SUPERUSER", "postgres")
    superpass = os.getenv("PG_SUPERPASSWORD")
    app_user = os.getenv("DB_USER", "tracex_user")

    print("=" * 65)
    print("TraceX PostGIS Setup & Geographic Column Provisioning")
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

        # 1. Enable PostGIS extension
        print("Enabling PostGIS extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("SELECT PostGIS_Version();")
        postgis_ver = cur.fetchone()[0]
        print(f"[OK] PostGIS Extension enabled! Version: {postgis_ver}")

        # 2. Add geography location column
        print("\nAdding 'location' geography column to 'vehicle_detections'...")
        cur.execute("""
            ALTER TABLE vehicle_detections 
            ADD COLUMN IF NOT EXISTS location geography(POINT, 4326);
        """)
        print("[OK] Column 'location geography(POINT, 4326)' verified.")

        # 3. Populate existing rows if latitude & longitude are present
        print("Populating 'location' from latitude/longitude for any existing rows...")
        cur.execute("""
            UPDATE vehicle_detections 
            SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL 
              AND location IS NULL;
        """)
        print(f"[OK] Rows updated: {cur.rowcount}")

        # 4. Create GiST spatial index
        print("Creating GiST spatial index 'idx_vehicle_detections_location'...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_detections_location 
            ON vehicle_detections USING GIST (location);
        """)
        print("[OK] GiST spatial index verified.")

        # 5. Grant permissions to tracex_user
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON TABLE vehicle_detections TO {};").format(sql.Identifier(app_user)))
        cur.execute(sql.SQL("GRANT SELECT ON TABLE spatial_ref_sys TO {};").format(sql.Identifier(app_user)))
        print(f"[OK] Granted privileges to user '{app_user}'.")

        print("\n" + "=" * 65)
        print("[SUCCESS] PostGIS integration and spatial schema setup completed successfully!")
        print("=" * 65)
        return True

    except Exception as e:
        print(f"\n[FAIL] PostGIS setup encountered an error:\n{e}")
        return False
    finally:
        if conn is not None and not conn.closed:
            conn.close()

if __name__ == "__main__":
    success = enable_postgis()
    sys.exit(0 if success else 1)
