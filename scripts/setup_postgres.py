"""
TraceX SIH 2026 - PostgreSQL Setup & Schema Migration Script
Creates the tracex database, tracex_user, vehicle_detections table, and required indexes.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database_and_schema():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    app_user = os.getenv("DB_USER", "tracex_user")
    app_password = os.getenv("DB_PASSWORD", "tracex_secure_password")

    pg_superuser = os.getenv("PG_SUPERUSER", "postgres")
    pg_superpassword = os.getenv("PG_SUPERPASSWORD", "postgres")

    print("=" * 65)
    print("TraceX PostgreSQL Setup & Schema Initialization")
    print("=" * 65)

    # 1. Connect as superuser to check/create user & database
    try:
        print(f"Connecting to default 'postgres' database on {host}:{port} as superuser '{pg_superuser}'...")
        admin_conn = psycopg2.connect(
            host=host,
            port=port,
            dbname="postgres",
            user=pg_superuser,
            password=pg_superpassword,
            connect_timeout=5
        )
        admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with admin_conn.cursor() as cur:
            # Check/create user
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (app_user,))
            if not cur.fetchone():
                print(f"Creating role/user '{app_user}'...")
                cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(sql.Identifier(app_user)), [app_password])
                print(f"[OK] User '{app_user}' created.")
            else:
                print(f"[OK] User '{app_user}' already exists.")

            # Check/create database
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
            if not cur.fetchone():
                print(f"Creating database '{dbname}' owned by '{app_user}'...")
                cur.execute(sql.SQL("CREATE DATABASE {} OWNER {};").format(sql.Identifier(dbname), sql.Identifier(app_user)))
                print(f"[OK] Database '{dbname}' created.")
            else:
                print(f"[OK] Database '{dbname}' already exists.")

            # Grant all privileges
            cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {};").format(sql.Identifier(dbname), sql.Identifier(app_user)))
        admin_conn.close()
    except Exception as e:
        print(f"[WARN] Superuser provisioning step encountered: {e}")
        print("Will attempt direct connection to target database...")

    # 2. Connect to tracex database to create schema and indexes
    try:
        print(f"\nConnecting to '{dbname}' database...")
        # Try connecting with app_user or fallback to superuser
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=app_user,
                password=app_password,
                connect_timeout=5
            )
        except Exception:
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=pg_superuser,
                password=pg_superpassword,
                connect_timeout=5
            )

        with conn.cursor() as cur:
            # Create vehicle_detections table
            print("Creating table 'vehicle_detections' if not exists...")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id BIGSERIAL PRIMARY KEY,
                plate_text VARCHAR(32) NOT NULL,
                cleaned_plate VARCHAR(20) NOT NULL,
                yolo_confidence DOUBLE PRECISION NOT NULL,
                ocr_confidence DOUBLE PRECISION NOT NULL,
                valid_indian_plate BOOLEAN NOT NULL DEFAULT FALSE,
                camera_id VARCHAR(64) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                frame_number INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """)

            # Create required indexes
            print("Creating indexes on 'vehicle_detections'...")
            indexes = [
                ("idx_vehicle_detections_cleaned_plate", "CREATE INDEX IF NOT EXISTS idx_vehicle_detections_cleaned_plate ON vehicle_detections(cleaned_plate);"),
                ("idx_vehicle_detections_timestamp", "CREATE INDEX IF NOT EXISTS idx_vehicle_detections_timestamp ON vehicle_detections(timestamp);"),
                ("idx_vehicle_detections_camera_id", "CREATE INDEX IF NOT EXISTS idx_vehicle_detections_camera_id ON vehicle_detections(camera_id);"),
                ("idx_vehicle_detections_track_id", "CREATE INDEX IF NOT EXISTS idx_vehicle_detections_track_id ON vehicle_detections(track_id);"),
            ]
            for idx_name, idx_sql in indexes:
                cur.execute(idx_sql)
                print(f"[OK] Index '{idx_name}' verified.")

            # Ensure permissions on table and sequences for app_user
            cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON TABLE vehicle_detections TO {};").format(sql.Identifier(app_user)))
            cur.execute(sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {};").format(sql.Identifier(app_user)))

        conn.commit()
        conn.close()
        print("\n[SUCCESS] PostgreSQL database and schema initialization completed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Schema creation failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_database_and_schema()
    sys.exit(0 if success else 1)
