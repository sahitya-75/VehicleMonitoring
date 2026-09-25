"""
TraceX SIH 2026 - PostgreSQL Database Connection Test
Verifies connection to the TraceX PostgreSQL database and fetches engine version.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

def test_connection():
    # Load environment variables from .env
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_secure_password")

    print("=" * 60)
    print("TraceX PostgreSQL Connection Test")
    print("=" * 60)
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
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version_str = cur.fetchone()[0]
            print("\n[SUCCESS] Connected successfully to PostgreSQL!")
            print(f"PostgreSQL Version: {version_str}")
        return True
    except psycopg2.OperationalError as e:
        print(f"\n[FAIL] Operational Error connecting to PostgreSQL:\n{e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during connection test:\n{e}")
        return False
    finally:
        if conn is not None and not conn.closed:
            conn.close()
            print("Connection closed cleanly.")
            print("=" * 60)

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
