import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DB_HOST", "localhost")
port = int(os.getenv("DB_PORT", "5432"))
dbname = os.getenv("DB_NAME", "tracex")
superuser = os.getenv("PG_SUPERUSER", "postgres")
superpass = os.getenv("PG_SUPERPASSWORD")

print(f"Connecting to {dbname} as {superuser}...")
conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=superuser,
    password=superpass,
    connect_timeout=5
)

with conn.cursor() as cur:
    cur.execute("SELECT name, default_version, installed_version, comment FROM pg_available_extensions WHERE name LIKE 'postgis%';")
    rows = cur.fetchall()
    print("Available PostGIS extensions in PostgreSQL 17:")
    if not rows:
        print("  NONE found.")
    for r in rows:
        print(f"  {r[0]}: default={r[1]}, installed={r[2]}")

conn.close()
