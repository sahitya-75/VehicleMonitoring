"""
TraceX SIH 2026 - Main FastAPI Application
Entry point for the TraceX ANPR & GIS Vehicle Monitoring REST API.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import psycopg2

# Ensure project root in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.routes.vehicles import router as vehicles_router
from api.routes.analytics import router as analytics_router
from api.routes.alerts import router as alerts_router

load_dotenv()

app = FastAPI(
    title="TraceX SIH 2026 - Vehicle Trajectory & Monitoring API",
    description="REST backend providing ANPR tracking, PostGIS spatial trajectories, traffic intelligence, rule engine violations, and vehicle watchlists.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include sub-routers
app.include_router(vehicles_router, prefix="/api/vehicles", tags=["Vehicles & Trajectories"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["Traffic Analytics"])
app.include_router(alerts_router, prefix="/api", tags=["Alerts & Watchlists"])

# Mount React frontend static files
from fastapi.staticfiles import StaticFiles
frontend_dir = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="dashboard")


@app.get("/api/health", tags=["System Health"])
def health_check() -> Dict[str, Any]:
    """
    Check operational status of the API, PostgreSQL database, and PostGIS extension.
    """
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "tracex")
    user = os.getenv("DB_USER", "tracex_user")
    password = os.getenv("DB_PASSWORD", "tracex_password_2026")

    db_status = "UNKNOWN"
    pg_version = "UNKNOWN"
    postgis_version = "UNKNOWN"
    record_count = 0

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=3
        )
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0]

            cur.execute("SELECT PostGIS_Version();")
            postgis_version = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM vehicle_detections;")
            record_count = cur.fetchone()[0]

        conn.close()
        db_status = "CONNECTED"
    except Exception as e:
        db_status = f"ERROR: {str(e)}"

    is_healthy = (db_status == "CONNECTED")

    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "status": db_status,
            "engine": "PostgreSQL",
            "version": pg_version,
            "postgis_version": postgis_version,
            "total_detections_indexed": record_count
        }
    }


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "TraceX SIH 2026",
        "service": "Vehicle Monitoring & Trajectory Reconstruction REST API",
        "documentation": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
