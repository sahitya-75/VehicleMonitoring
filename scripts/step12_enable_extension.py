"""
TraceX SIH 2026 - STEP 12.3: Enable PostGIS Extension
Enables PostGIS extension in 'tracex' database and verifies with PostGIS_Version().
Does NOT modify vehicle_detections table or create spatial indexes.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

def enable_postgis_extension():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    superuser = os.getenv("PG_SUPERUSER", "postgres")
    superpass = os.getenv("PG_SUPERPASSWORD")
    app_user = os.getenv("DB_USER", "tracex_user")

    print("=" * 60)
    print("TraceX STEP 12.3 - Enable PostGIS Extension")
    print("=" * 60)
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
        print("Executing: CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

        # 2. Verify PostGIS version
        cur.execute("SELECT PostGIS_Version();")
        version_result = cur.fetchone()[0]
        print(f"\n[SUCCESS] PostGIS Version: {version_result}")

        # Grant SELECT on spatial_ref_sys to tracex_user so tracex_user can use geography functions
        cur.execute(sql.SQL("GRANT SELECT ON TABLE spatial_ref_sys TO {};").format(sql.Identifier(app_user)))
        print(f"[OK] Granted SELECT on spatial_ref_sys to '{app_user}'.")

        print("=" * 60)
        return True, version_result, None

    except Exception as e:
        print(f"\n[FAIL] Error enabling PostGIS extension:\n{e}")
        print("=" * 60)
        return False, None, str(e)
    finally:
        if conn is not None and not conn.closed:
            conn.close()

if __name__ == "__main__":
    success, ver, err = enable_postgis_extension()
    sys.exit(0 if success else 1)
