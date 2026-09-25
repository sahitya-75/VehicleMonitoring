"""
TraceX SIH 2026 - Traffic Analytics API Routes
Exposes traffic intelligence, camera statistics, hourly distributions, and corridor patterns.
"""

from typing import Dict, List, Any
from fastapi import APIRouter
from analytics.traffic_analytics import TrafficAnalyticsEngine

router = APIRouter()
analytics_engine = TrafficAnalyticsEngine()


@router.get("/summary")
def get_analytics_summary() -> Dict[str, Any]:
    """
    Return comprehensive traffic analytics summary including speeds, routes, and totals.
    """
    return analytics_engine.generate_full_analytics_report()


@router.get("/cameras")
def get_camera_analytics() -> List[Dict[str, Any]]:
    """
    Return camera-wise traffic flow, unique vehicle counts, and sensor confidence metrics.
    """
    return analytics_engine.get_camera_traffic_stats()


@router.get("/hourly")
def get_hourly_analytics() -> List[Dict[str, Any]]:
    """
    Return hourly traffic volume distribution throughout the day.
    """
    return analytics_engine.get_hourly_traffic_distribution()
