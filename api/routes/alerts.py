"""
TraceX SIH 2026 - Alerts & Watchlist API Routes
Provides endpoints for monitoring alerts, managing vehicle watchlists, and triggering scans.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from alerts.alert_service import AlertService, WatchlistItem, AlertRecord

router = APIRouter()
alert_service = AlertService()


class WatchlistCreateRequest(BaseModel):
    cleaned_plate: str = Field(..., min_length=2, max_length=20, description="Vehicle license plate to monitor")
    description: str = Field(..., min_length=3, description="Reason or case reference for monitoring")
    priority: str = Field("HIGH", description="Alert priority (LOW, MEDIUM, HIGH, CRITICAL)")
    enabled: bool = Field(True, description="Whether watchlist item is active")


class WatchlistResponse(BaseModel):
    cleaned_plate: str
    description: str
    priority: str
    enabled: bool
    created_at: str


class AlertResponse(BaseModel):
    alert_id: str
    plate: str
    alert_type: str
    severity: str
    camera_id: str
    timestamp: str
    message: str
    source_rule_id: Optional[str] = None
    detection_id: Optional[int] = None
    created_at: str


@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    alert_type: Optional[str] = Query(None, description="Filter by alert type (e.g. WATCHLIST_HIT, SPEEDING_VIOLATION)"),
    severity: Optional[str] = Query(None, description="Filter by severity (e.g. HIGH, CRITICAL, MEDIUM)"),
    plate: Optional[str] = Query(None, description="Filter by plate number"),
    refresh: bool = Query(True, description="Auto-scan latest detections and rule findings before returning")
):
    """
    Return all generated alerts, with optional automatic scan and filtering.
    """
    if refresh:
        alert_service.scan_watchlist_hits()
        alert_service.process_rule_findings()

    alerts = alert_service.get_all_alerts(
        alert_type=alert_type,
        severity=severity,
        plate=plate
    )
    return [a.to_dict() for a in alerts]


@router.get("/watchlist", response_model=List[WatchlistResponse])
def get_watchlist(
    only_enabled: bool = Query(False, description="Return only active/enabled watchlist items")
):
    """
    List all vehicle watchlist entries.
    """
    items = alert_service.list_watchlist(only_enabled=only_enabled)
    return [item.to_dict() for item in items]


@router.post("/watchlist", response_model=WatchlistResponse, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(req: WatchlistCreateRequest):
    """
    Add or update a vehicle on the alert watchlist.
    """
    cleaned = req.cleaned_plate.strip().upper().replace(" ", "").replace("-", "")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid plate number provided.")

    item = alert_service.add_to_watchlist(
        plate=cleaned,
        description=req.description,
        priority=req.priority.upper(),
        enabled=req.enabled
    )

    # Immediately trigger a scan for the newly added vehicle
    alert_service.scan_watchlist_hits()

    return item.to_dict()


@router.delete("/watchlist/{plate}")
def delete_from_watchlist(plate: str):
    """
    Remove a vehicle from the alert watchlist.
    """
    cleaned = plate.strip().upper().replace(" ", "").replace("-", "")
    removed = alert_service.remove_from_watchlist(cleaned)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Vehicle '{plate}' not found in watchlist."
        )

    return {
        "success": True,
        "message": f"Vehicle '{cleaned}' removed from watchlist."
    }
