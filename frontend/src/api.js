/**
 * TraceX SIH 2026 - Centralized Frontend API Client
 * Interfaces directly with the FastAPI backend at http://127.0.0.1:8000
 */

const API_BASE = "http://127.0.0.1:8000/api";

export async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });

    if (!res.ok) {
      let errMsg = `HTTP ${res.status} ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      throw new Error(errMsg);
    }

    return await res.json();
  } catch (error) {
    console.error(`[API Error] ${endpoint}:`, error);
    throw error;
  }
}

// System Health
export const getHealth = () => fetchJson("/health");

// Vehicles & Trajectories
export const getVehicles = (page = 1, limit = 20, camera_id = null, valid_only = false, search = null) => {
  const params = new URLSearchParams({ page, limit, valid_only });
  if (camera_id) params.append("camera_id", camera_id);
  if (search) params.append("search", search);
  return fetchJson(`/vehicles?${params.toString()}`);
};

export const getVehicleSummary = (plate) => fetchJson(`/vehicles/${encodeURIComponent(plate)}`);

export const getVehicleTrajectory = (plate) => fetchJson(`/vehicles/${encodeURIComponent(plate)}/trajectory`);

export const searchVehicles = (plate, fuzzy = true, max_dist = 2) => {
  const params = new URLSearchParams({ plate, fuzzy, max_dist });
  return fetchJson(`/vehicles/search?${params.toString()}`);
};

// Analytics
export const getAnalyticsSummary = () => fetchJson("/analytics/summary");
export const getAnalyticsCameras = () => fetchJson("/analytics/cameras");
export const getAnalyticsHourly = () => fetchJson("/analytics/hourly");

// Alerts & Watchlist
export const getAlerts = (alert_type = null, severity = null, plate = null) => {
  const params = new URLSearchParams();
  if (alert_type) params.append("alert_type", alert_type);
  if (severity) params.append("severity", severity);
  if (plate) params.append("plate", plate);
  return fetchJson(`/alerts?${params.toString()}`);
};

export const getWatchlist = (only_enabled = false) => {
  return fetchJson(`/watchlist?only_enabled=${only_enabled}`);
};

export const addToWatchlist = (data) => {
  return fetchJson("/watchlist", {
    method: "POST",
    body: JSON.stringify(data)
  });
};

export const deleteFromWatchlist = (plate) => {
  return fetchJson(`/watchlist/${encodeURIComponent(plate)}`, {
    method: "DELETE"
  });
};
