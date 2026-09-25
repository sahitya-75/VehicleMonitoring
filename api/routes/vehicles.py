"""
TraceX SIH 2026 - Vehicles & Trajectory API Routes
Provides endpoints for querying detection records, vehicle sightings, and spatial trajectories.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from trajectory.trajectory_service import TrajectoryService, clean_plate_string

router = APIRouter()
trajectory_service = TrajectoryService()


class TrajectoryPointModel(BaseModel):
    id: int
    camera_id: str
    timestamp: str
    frame_number: int
    track_id: int
    latitude: float
    longitude: float
    plate_text: str
    cleaned_plate: str
    yolo_confidence: float
    ocr_confidence: float
    valid_indian_plate: bool
    distance_from_prev_m: float = 0.0
    time_delta_seconds: float = 0.0
    speed_kmh: float = 0.0


class VehicleTrajectoryResponse(BaseModel):
    searched_plate: str
    matched_plate: str
    match_type: str
    edit_distance: int
    total_sightings: int
    first_seen: str
    last_seen: str
    total_distance_meters: float
    unique_cameras: List[str]
    points: List[TrajectoryPointModel]


class VehicleListResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    records: List[Dict[str, Any]]


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[VehicleTrajectoryResponse]


@router.get("", response_model=VehicleListResponse)
def get_vehicles(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Records per page"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    valid_only: bool = Query(False, description="Filter valid Indian plates only"),
    search: Optional[str] = Query(None, description="Filter by plate search substring")
):
    """
    Return paginated vehicle detection records from PostgreSQL database.
    """
    offset = (page - 1) * limit
    where_clauses = []
    params = []

    if camera_id:
        where_clauses.append("camera_id = %s")
        params.append(camera_id)
    if valid_only:
        where_clauses.append("valid_indian_plate = TRUE")
    if search:
        cleaned_search = clean_plate_string(search)
        where_clauses.append("cleaned_plate ILIKE %s")
        params.append(f"%{cleaned_search}%")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = trajectory_service._get_connection()
    try:
        with conn.cursor() as cur:
            # Count query
            cur.execute(f"SELECT COUNT(*) FROM vehicle_detections {where_sql};", tuple(params))
            total_count = cur.fetchone()[0]

            # Fetch page
            cur.execute(f"""
                SELECT 
                    id, plate_text, cleaned_plate, yolo_confidence, ocr_confidence,
                    valid_indian_plate, camera_id, timestamp, frame_number, track_id,
                    latitude, longitude
                FROM vehicle_detections
                {where_sql}
                ORDER BY timestamp DESC, frame_number DESC
                LIMIT %s OFFSET %s;
            """, tuple(params + [limit, offset]))

            cols = [desc[0] for desc in cur.description]
            records = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                if hasattr(d.get("timestamp"), "isoformat"):
                    d["timestamp"] = d["timestamp"].isoformat()
                records.append(d)

        pages = (total_count + limit - 1) // limit if total_count > 0 else 0
        return {
            "total": total_count,
            "page": page,
            "limit": limit,
            "pages": pages,
            "records": records
        }
    finally:
        conn.close()


@router.get("/search", response_model=SearchResponse)
def search_vehicles(
    plate: str = Query(..., min_length=1, description="Plate number to search"),
    fuzzy: bool = Query(True, description="Allow fuzzy Levenshtein matching"),
    max_dist: int = Query(2, ge=0, le=5, description="Maximum edit distance for fuzzy search")
):
    """
    Search vehicle trajectories supporting exact and fuzzy OCR variation matching.
    """
    cleaned = clean_plate_string(plate)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Plate parameter must contain alphanumeric characters.")

    trajectories = trajectory_service.query_trajectory(
        plate=cleaned,
        allow_fuzzy=fuzzy,
        max_edit_distance=max_dist
    )

    return {
        "query": plate,
        "count": len(trajectories),
        "results": [t.to_dict() for t in trajectories]
    }


@router.get("/{plate}/trajectory", response_model=VehicleTrajectoryResponse)
def get_vehicle_trajectory(plate: str):
    """
    Return the complete, ordered chronological trajectory for a specific vehicle plate.
    """
    cleaned = clean_plate_string(plate)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid plate number provided.")

    traj = trajectory_service.get_exact_trajectory(cleaned)
    if not traj:
        # Fallback to single closest fuzzy match if exact is not found
        fuzzy_matches = trajectory_service.query_trajectory(cleaned, allow_fuzzy=True, max_edit_distance=1)
        if fuzzy_matches:
            traj = fuzzy_matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"No trajectory found for plate '{plate}'.")

    return traj.to_dict()


@router.get("/{plate}")
def get_vehicle_summary(plate: str):
    """
    Return chronological vehicle sightings and summary metrics for a plate.
    """
    cleaned = clean_plate_string(plate)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid plate number provided.")

    traj = trajectory_service.get_exact_trajectory(cleaned)
    if not traj:
        raise HTTPException(status_code=404, detail=f"Vehicle '{plate}' not found.")

    return {
        "plate": traj.matched_plate,
        "total_sightings": traj.total_sightings,
        "first_seen": traj.first_seen,
        "last_seen": traj.last_seen,
        "total_distance_km": round(traj.total_distance_meters / 1000.0, 2),
        "cameras_visited": traj.unique_cameras,
        "sightings": [p.to_dict() for p in traj.points]
    }
