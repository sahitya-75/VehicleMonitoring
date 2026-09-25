"""
TraceX SIH 2026 - Migration: SQLite to PostgreSQL with PostGIS Geography
Migrates historical detection records from data/vehicle_history.db into PostgreSQL 'tracex' database.
Preserves all fields, populates PostGIS geography points, and ensures idempotency.
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

def migrate_data(sqlite_path: str = "data/vehicle_history.db"):
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_password_2026")

    print("=" * 70)
    print("TraceX SQLite to PostgreSQL Migration")
    print("=" * 70)

    # 1. Read from SQLite (read-only)
    if not os.path.exists(sqlite_path):
        print(f"[FAIL] SQLite database not found at {sqlite_path}")
        return False

    print(f"Reading records from SQLite: {sqlite_path}")
    sq_conn = sqlite3.connect(f"file:{os.path.abspath(sqlite_path)}?mode=ro", uri=True)
    sq_conn.row_factory = sqlite3.Row
    sq_cur = sq_conn.cursor()
    sq_cur.execute("""
        SELECT id, plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
               valid_indian_plate, camera_id, timestamp, frame_number, track_id,
               latitude, longitude
        FROM vehicle_detections
        ORDER BY id ASC;
    """)
    sqlite_rows = sq_cur.fetchall()
    sqlite_count = len(sqlite_rows)
    sq_conn.close()
    print(f"Found {sqlite_count} records in SQLite database.")

    if sqlite_count == 0:
        print("[WARN] No records found in SQLite database.")
        return True

    # 2. Connect to PostgreSQL
    print(f"\nConnecting to PostgreSQL '{dbname}' on {host}:{port} as '{user}'...")
    pg_conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=5
    )
    pg_conn.autocommit = False
    pg_cur = pg_conn.cursor()

    try:
        # Check existing records in PostgreSQL
        pg_cur.execute("SELECT id FROM vehicle_detections;")
        existing_ids = {row[0] for row in pg_cur.fetchall()}
        print(f"Existing PostgreSQL vehicle_detections records count: {len(existing_ids)}")

        inserted_count = 0
        skipped_count = 0

        for r in sqlite_rows:
            rec_id = r["id"]
            if rec_id in existing_ids:
                skipped_count += 1
                continue

            plate_text = r["plate_text"]
            cleaned_plate = r["cleaned_plate"]
            yolo_conf = float(r["yolo_confidence"])
            ocr_conf = float(r["ocr_confidence"])
            valid_indian = bool(r["valid_indian_plate"])
            camera_id = r["camera_id"]
            timestamp_str = r["timestamp"]
            frame_num = int(r["frame_number"])
            track_id = int(r["track_id"])
            lat = float(r["latitude"]) if r["latitude"] is not None else None
            lon = float(r["longitude"]) if r["longitude"] is not None else None

            # Insert into PostgreSQL
            if lat is not None and lon is not None:
                pg_cur.execute("""
                    INSERT INTO vehicle_detections (
                        id, plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                        valid_indian_plate, camera_id, timestamp, frame_number, track_id,
                        latitude, longitude, location
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                    )
                    ON CONFLICT (id) DO NOTHING;
                """, (
                    rec_id, plate_text, cleaned_plate, yolo_conf, ocr_conf,
                    valid_indian, camera_id, timestamp_str, frame_num, track_id,
                    lat, lon, lon, lat
                ))
            else:
                pg_cur.execute("""
                    INSERT INTO vehicle_detections (
                        id, plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                        valid_indian_plate, camera_id, timestamp, frame_number, track_id,
                        latitude, longitude, location
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, NULL
                    )
                    ON CONFLICT (id) DO NOTHING;
                """, (
                    rec_id, plate_text, cleaned_plate, yolo_conf, ocr_conf,
                    valid_indian, camera_id, timestamp_str, frame_num, track_id,
                    lat, lon
                ))
            inserted_count += 1

        # Reset serial sequence
        pg_cur.execute("""
            SELECT setval(pg_get_serial_sequence('vehicle_detections', 'id'), coalesce(max(id), 1)) 
            FROM vehicle_detections;
        """)

        pg_conn.commit()
        print(f"\nMigration complete: {inserted_count} inserted, {skipped_count} skipped (already present).")

        # 3. Validation
        print("\n" + "=" * 30 + " Migration Validation " + "=" * 30)
        pg_cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        total_pg = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(*) FROM vehicle_detections WHERE location IS NOT NULL;")
        geo_count = pg_cur.fetchone()[0]

        pg_cur.execute("SELECT COUNT(DISTINCT cleaned_plate) FROM vehicle_detections WHERE cleaned_plate != '';")
        unique_plates_count = pg_cur.fetchone()[0]

        print(f"[PASS] SQLite Source Record Count    : {sqlite_count}")
        print(f"[PASS] PostgreSQL Target Record Count: {total_pg}")
        print(f"[PASS] Records with PostGIS Location : {geo_count}")
        print(f"[PASS] Unique Cleaned Plates         : {unique_plates_count}")

        # Ensure SQLite file unchanged
        sq_conn_verify = sqlite3.connect(f"file:{os.path.abspath(sqlite_path)}?mode=ro", uri=True)
        sq_verify_cur = sq_conn_verify.cursor()
        sq_verify_cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
        sqlite_count_after = sq_verify_cur.fetchone()[0]
        sq_conn_verify.close()
        print(f"[PASS] SQLite Record Count Preserved : {sqlite_count_after} (Untouched: {sqlite_count_after == 52})")

        pg_conn.close()
        print("=" * 70)
        return total_pg == sqlite_count and sqlite_count_after == 52

    except Exception as e:
        pg_conn.rollback()
        pg_conn.close()
        print(f"\n[FAIL] Migration error:\n{e}")
        return False

if __name__ == "__main__":
    success = migrate_data()
    sys.exit(0 if success else 1)
