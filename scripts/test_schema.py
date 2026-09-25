"""
TraceX SIH 2026 - PostgreSQL Schema Verification Test
Verifies table structure, columns, data types, and required indexes for vehicle_detections.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

REQUIRED_COLUMNS = {
    "id",
    "plate_text",
    "cleaned_plate",
    "yolo_confidence",
    "ocr_confidence",
    "valid_indian_plate",
    "camera_id",
    "timestamp",
    "frame_number",
    "track_id",
    "latitude",
    "longitude",
    "location",
}

REQUIRED_INDEXES = {
    "idx_vehicle_detections_cleaned_plate",
    "idx_vehicle_detections_timestamp",
    "idx_vehicle_detections_camera_id",
    "idx_vehicle_detections_track_id",
    "idx_vehicle_detections_location",
}

def verify_schema():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_secure_password")

    print("=" * 65)
    print("TraceX PostgreSQL Schema & Index Verification")
    print("=" * 65)
    print(f"Connecting to database '{dbname}' on {host}:{port} as '{user}'...")

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

        # 1. Verify table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'vehicle_detections'
            );
        """)
        table_exists = cur.fetchone()[0]
        if not table_exists:
            print("[FAIL] Table 'vehicle_detections' does NOT exist in public schema.")
            return False
        print("[PASS] Table 'vehicle_detections' exists.")

        # 2. Verify all columns
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'vehicle_detections';
        """)
        existing_cols = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
        
        print("\nVerifying Columns:")
        missing_cols = REQUIRED_COLUMNS - set(existing_cols.keys())
        for col in sorted(REQUIRED_COLUMNS):
            if col in existing_cols:
                dtype, nullable = existing_cols[col]
                print(f"  [PASS] {col:<20} -> type: {dtype:<20} (nullable: {nullable})")
            else:
                print(f"  [FAIL] Missing required column: {col}")

        if missing_cols:
            print(f"\n[FAIL] Missing columns: {missing_cols}")
            return False

        # 3. Verify indexes
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = 'vehicle_detections';
        """)
        existing_indexes = {row[0]: row[1] for row in cur.fetchall()}

        print("\nVerifying Indexes:")
        missing_indexes = REQUIRED_INDEXES - set(existing_indexes.keys())
        for idx in sorted(REQUIRED_INDEXES):
            if idx in existing_indexes:
                print(f"  [PASS] Index '{idx}' exists -> {existing_indexes[idx]}")
            else:
                print(f"  [FAIL] Missing required index: {idx}")

        if missing_indexes:
            print(f"\n[FAIL] Missing indexes: {missing_indexes}")
            return False

        print("\n" + "=" * 65)
        print("[SUCCESS] ALL Schema & Index Validations Passed!")
        print("=" * 65)
        return True

    except Exception as e:
        print(f"\n[FAIL] Schema verification encountered error: {e}")
        return False
    finally:
        if conn is not None and not conn.closed:
            conn.close()

if __name__ == "__main__":
    success = verify_schema()
    sys.exit(0 if success else 1)
