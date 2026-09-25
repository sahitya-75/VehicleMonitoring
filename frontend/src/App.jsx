const { useState, useEffect, useMemo, useRef } = React;
import * as api from './api.js';

// =========================================================================
// Top Navigation Header
// =========================================================================
function TopHeader({ activeTab, onSearchPlate, health }) {
  const [time, setTime] = useState(new Date().toLocaleTimeString());
  const [searchInput, setSearchInput] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleSearch = (e) => {
    if (e.key === 'Enter' && searchInput.trim()) {
      onSearchPlate(searchInput.trim().toUpperCase());
      setSearchInput('');
    }
  };

  const titles = {
    overview: 'Operational Overview & Surveillance Nodes',
    vehicles: 'ANPR Vehicle Intelligence & Sighting History',
    trajectory: 'GIS Spatial Trajectory Reconstruction',
    analytics: 'Corridor Traffic Analytics & Speed Intelligence',
    alerts: 'Infraction Detection & Live Alerts Feed',
    watchlist: 'Vehicle Watchlist & Target Interception'
  };

  const isHealthy = health?.status === 'HEALTHY';

  return (
    <header className="top-header">
      <div className="header-title">
        <h1>{titles[activeTab] || 'Dashboard'}</h1>
      </div>
      <div className="header-meta">
        <div className="quick-search-box">
          <span style={{ fontSize: '12px' }}>🔍</span>
          <input
            type="text"
            placeholder="QUICK PLATE..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={handleSearch}
          />
        </div>

        <div className="system-status">
          <div className="status-dot" style={{ backgroundColor: isHealthy ? '#10b981' : '#f43f5e' }}></div>
          <span>{isHealthy ? 'POSTGIS 3.6 / PG 17' : 'OFFLINE'}</span>
        </div>

        <div className="clock-widget">
          {time} IST
        </div>
      </div>
    </header>
  );
}

// =========================================================================
// Sidebar Navigation
// =========================================================================
function Sidebar({ activeTab, setActiveTab, alertCount }) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'vehicles', label: 'Vehicle Search', icon: '🔍' },
    { id: 'trajectory', label: 'Trajectory GIS Map', icon: '🛰️' },
    { id: 'analytics', label: 'Traffic Analytics', icon: '📈' },
    { id: 'alerts', label: 'Alerts & Violations', icon: '🚨', badge: alertCount },
    { id: 'watchlist', label: 'Watchlist', icon: '🎯' }
  ];

  return (
    <aside className="sidebar">
      <div>
        <div className="brand">
          <div className="brand-icon">TX</div>
          <div className="brand-info">
            <h2>TraceX</h2>
            <div className="brand-badge">SIH 2026 Core</div>
          </div>
        </div>

        <ul className="nav-menu">
          {navItems.map((item) => (
            <li
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span className="icon">{item.icon}</span>
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.badge ? (
                <span className="badge critical" style={{ fontSize: '9px', padding: '1px 6px' }}>
                  {item.badge}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar-footer">
        <div style={{ fontSize: '11px', color: 'var(--text-sub)', textAlign: 'center' }}>
          TraceX ANPR Engine v1.0<br/>PostGIS Spatial Layer
        </div>
      </div>
    </aside>
  );
}

// =========================================================================
// Component: Overview Dashboard
// =========================================================================
function OverviewView({ onSelectPlate, onSwitchTab }) {
  const [summary, setSummary] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [recentDetections, setRecentDetections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [sumData, camData, vehData] = await Promise.all([
          api.getAnalyticsSummary(),
          api.getAnalyticsCameras(),
          api.getVehicles(1, 8)
        ]);
        setSummary(sumData);
        setCameras(camData);
        setRecentDetections(vehData.records || []);
      } catch (err) {
        console.error('Overview load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="loading-box">
        <div className="spinner"></div>
        <p>Loading System Telemetry & Metrics...</p>
      </div>
    );
  }

  const s = summary?.summary || {};
  const sp = summary?.speed_and_distance || {};

  return (
    <div>
      {/* 8 Primary Metrics Cards */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-card-header">
            <span>Total Detections</span>
            <span>📹</span>
          </div>
          <div className="metric-card-value">{s.total_detections || 52}</div>
          <div className="metric-card-sub">All camera passes recorded</div>
        </div>

        <div className="metric-card green">
          <div className="metric-card-header">
            <span>Unique Vehicles</span>
            <span>🚗</span>
          </div>
          <div className="metric-card-value">{s.unique_vehicles || 24}</div>
          <div className="metric-card-sub">Distinct vehicle tracks</div>
        </div>

        <div className="metric-card purple">
          <div className="metric-card-header">
            <span>Unique Plates</span>
            <span>🏷️</span>
          </div>
          <div className="metric-card-value">{s.unique_plates || 24}</div>
          <div className="metric-card-sub">{s.valid_indian_plates || 12} valid Indian formats</div>
        </div>

        <div className="metric-card amber">
          <div className="metric-card-header">
            <span>Active Cameras</span>
            <span>📡</span>
          </div>
          <div className="metric-card-value">{cameras.length || 5}</div>
          <div className="metric-card-sub">Surveillance stations</div>
        </div>

        <div className="metric-card rose">
          <div className="metric-card-header">
            <span>Traffic Violations</span>
            <span>⚠️</span>
          </div>
          <div className="metric-card-value">12</div>
          <div className="metric-card-sub">4 Speeding | 8 Repeated loops</div>
        </div>

        <div className="metric-card rose">
          <div className="metric-card-header">
            <span>Active Alerts</span>
            <span>🚨</span>
          </div>
          <div className="metric-card-value">16</div>
          <div className="metric-card-sub">4 Watchlist hits | 12 Rule alerts</div>
        </div>

        <div className="metric-card green">
          <div className="metric-card-header">
            <span>Avg Transit Speed</span>
            <span>⚡</span>
          </div>
          <div className="metric-card-value">{sp.average_speed_kmh || 41.08} <span style={{ fontSize: '14px' }}>km/h</span></div>
          <div className="metric-card-sub">Max: {sp.max_speed_kmh || 56.04} km/h</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span>Monitored Corridor</span>
            <span>🗺️</span>
          </div>
          <div className="metric-card-value">39.66 <span style={{ fontSize: '14px' }}>km</span></div>
          <div className="metric-card-sub">Delhi North to Noida Expressway</div>
        </div>
      </div>

      {/* Grid: Cameras Load & Recent Detections */}
      <div className="dashboard-grid">
        <div className="glass-panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>📡</span> Camera Sensor Nodes & Traffic Breakdown
            </div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Camera Station</th>
                <th>Detections</th>
                <th>Vehicles</th>
                <th>Plates</th>
                <th>Avg YOLO</th>
                <th>Avg OCR</th>
              </tr>
            </thead>
            <tbody>
              {cameras.map((cam) => (
                <tr key={cam.camera_id}>
                  <td className="mono" style={{ color: '#fff' }}>{cam.camera_id}</td>
                  <td><span className="badge medium">{cam.detection_count}</span></td>
                  <td>{cam.vehicle_count}</td>
                  <td>{cam.unique_plates_count}</td>
                  <td>{(cam.avg_yolo_confidence * 100).toFixed(1)}%</td>
                  <td>{(cam.avg_ocr_confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="glass-panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>⚡</span> Recent ANPR Detections
            </div>
            <button className="btn" style={{ padding: '4px 10px', fontSize: '11px' }} onClick={() => onSwitchTab('vehicles')}>
              View All
            </button>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Plate</th>
                <th>Camera</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {recentDetections.map((r) => (
                <tr key={r.id}>
                  <td className="mono" style={{ color: 'var(--accent-blue)' }}>
                    {r.cleaned_plate || r.plate_text || 'UNREAD'}
                  </td>
                  <td style={{ fontSize: '11px' }}>{r.camera_id}</td>
                  <td>
                    <button
                      className="btn"
                      style={{ padding: '2px 8px', fontSize: '10px' }}
                      onClick={() => onSelectPlate(r.cleaned_plate || r.plate_text)}
                    >
                      Track
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// Component: Vehicle Search & Sighting History
// =========================================================================
function VehicleSearchView({ selectedPlate, onSelectPlate, onTrackOnMap }) {
  const [searchQuery, setSearchQuery] = useState(selectedPlate || 'HR51BC3493');
  const [fuzzy, setFuzzy] = useState(true);
  const [maxDist, setMaxDist] = useState(2);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    if (selectedPlate) {
      setSearchQuery(selectedPlate);
      performSearch(selectedPlate);
    }
  }, [selectedPlate]);

  const performSearch = async (plate) => {
    if (!plate) return;
    setLoading(true);
    setSearched(true);
    try {
      const data = await api.searchVehicles(plate, fuzzy, maxDist);
      setResults(data.results || []);
    } catch (err) {
      console.error('Search error:', err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const quickChips = ['HR51BC3493', 'DL3CC8387', 'UP21X3666', 'ZCL3285', 'HR51BC3498'];

  return (
    <div>
      <div className="glass-panel">
        <div className="panel-header">
          <div className="panel-title"><span>🔍</span> ANPR License Plate Search Engine</div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            className="form-control"
            style={{ width: '280px', fontFamily: 'JetBrains Mono', fontWeight: 'bold' }}
            value={searchQuery}
            placeholder="ENTER VEHICLE REGISTRATION..."
            onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === 'Enter' && performSearch(searchQuery)}
          />

          <button className="btn" onClick={() => performSearch(searchQuery)}>
            Search Database
          </button>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', cursor: 'pointer', color: 'var(--text-muted)' }}>
            <input
              type="checkbox"
              checked={fuzzy}
              onChange={(e) => setFuzzy(e.target.checked)}
            />
            Allow Fuzzy Match (Levenshtein Distance ≤ {maxDist})
          </label>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '14px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Suggestions:</span>
          {quickChips.map((chip) => (
            <button
              key={chip}
              className="btn"
              style={{ padding: '3px 8px', fontSize: '11px', background: 'rgba(255,255,255,0.06)' }}
              onClick={() => {
                setSearchQuery(chip);
                performSearch(chip);
              }}
            >
              {chip}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-box">
          <div className="spinner"></div>
          <p>Querying PostgreSQL & Trajectory Reconstructor...</p>
        </div>
      )}

      {!loading && searched && results.length === 0 && (
        <div className="empty-box glass-panel">
          <div style={{ fontSize: '32px', marginBottom: '8px' }}>🚫</div>
          <h3>No Vehicle Matches Found</h3>
          <p style={{ marginTop: '4px', color: 'var(--text-muted)', fontSize: '13px' }}>
            No records matched plate <strong>{searchQuery}</strong> within edit distance threshold.
          </p>
        </div>
      )}

      {!loading && results.map((traj) => (
        <div key={traj.matched_plate} className="glass-panel" style={{ marginTop: '16px' }}>
          <div className="panel-header">
            <div>
              <span className="mono" style={{ fontSize: '18px', color: '#fff', fontWeight: 'bold' }}>
                {traj.matched_plate}
              </span>
              <span className={`badge ${traj.match_type === 'EXACT' ? 'green' : 'high'}`} style={{ marginLeft: '10px' }}>
                {traj.match_type} (Dist: {traj.edit_distance})
              </span>
            </div>

            <button
              className="btn"
              onClick={() => onTrackOnMap(traj.matched_plate)}
            >
              🛰️ Track on GIS Map
            </button>
          </div>

          <div className="metrics-row" style={{ marginBottom: '16px' }}>
            <div className="metric-card">
              <div className="metric-card-header"><span>Sightings</span></div>
              <div className="metric-card-value">{traj.total_sightings}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card-header"><span>Total Distance</span></div>
              <div className="metric-card-value">{(traj.total_distance_meters / 1000).toFixed(2)} km</div>
            </div>
            <div className="metric-card">
              <div className="metric-card-header"><span>Cameras Visited</span></div>
              <div className="metric-card-value">{traj.unique_cameras.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card-header"><span>First Seen</span></div>
              <div className="metric-card-value" style={{ fontSize: '13px' }}>{new Date(traj.first_seen).toLocaleTimeString()}</div>
            </div>
          </div>

          <div className="panel-title" style={{ fontSize: '12px', marginBottom: '8px' }}>
            Chronological Waypoints ({traj.points.length} Sightings)
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Camera Station</th>
                <th>Timestamp</th>
                <th>Coordinates</th>
                <th>Leg Dist (m)</th>
                <th>Speed (km/h)</th>
                <th>OCR Confidence</th>
              </tr>
            </thead>
            <tbody>
              {traj.points.map((pt, idx) => (
                <tr key={pt.id}>
                  <td>{idx + 1}</td>
                  <td className="mono">{pt.camera_id}</td>
                  <td>{new Date(pt.timestamp).toLocaleTimeString()}</td>
                  <td className="mono">{pt.latitude.toFixed(4)}, {pt.longitude.toFixed(4)}</td>
                  <td>{pt.distance_from_prev_m.toFixed(1)} m</td>
                  <td>
                    {pt.speed_kmh > 50 ? (
                      <span className="badge critical">{pt.speed_kmh.toFixed(1)} km/h</span>
                    ) : (
                      <span>{pt.speed_kmh.toFixed(1)} km/h</span>
                    )}
                  </td>
                  <td>{(pt.ocr_confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

// =========================================================================
// Component: Trajectory GIS Map (Leaflet)
// =========================================================================
function TrajectoryMapView({ targetPlate = 'HR51BC3493', onSelectPlate }) {
  const [currentPlate, setCurrentPlate] = useState(targetPlate);
  const [trajData, setTrajData] = useState(null);
  const [loading, setLoading] = useState(false);
  const mapRef = useRef(null);
  const leafletInstance = useRef(null);
  const polylineLayer = useRef(null);
  const markersLayer = useRef(null);

  useEffect(() => {
    // Initialize Leaflet Map once
    if (!leafletInstance.current && document.getElementById('leaflet-map')) {
      const map = L.map('leaflet-map', { zoomControl: false }).setView([28.6139, 77.2090], 11);
      L.control.zoom({ position: 'topright' }).addTo(map);

      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CARTO &copy; OpenStreetMap',
        subdomains: 'abcd',
        maxZoom: 19
      }).addTo(map);

      polylineLayer.current = L.layerGroup().addTo(map);
      markersLayer.current = L.layerGroup().addTo(map);
      leafletInstance.current = map;
    }

    loadTrajectory(currentPlate);
  }, [currentPlate]);

  const loadTrajectory = async (plate) => {
    if (!plate) return;
    setLoading(true);
    try {
      const data = await api.getVehicleTrajectory(plate);
      setTrajData(data);
      renderMapFeatures(data);
    } catch (err) {
      console.error('Trajectory fetch error:', err);
      setTrajData(null);
    } finally {
      setLoading(false);
    }
  };

  const renderMapFeatures = (data) => {
    const map = leafletInstance.current;
    if (!map || !polylineLayer.current || !markersLayer.current) return;

    polylineLayer.current.clearLayers();
    markersLayer.current.clearLayers();

    const points = data?.points || [];
    if (points.length === 0) return;

    const latlngs = points.map((p) => [p.latitude, p.longitude]);

    // Draw Polyline
    if (latlngs.length > 1) {
      const poly = L.polyline(latlngs, {
        color: '#38bdf8',
        weight: 4,
        opacity: 0.9,
        dashArray: '8, 8',
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(polylineLayer.current);

      map.fitBounds(poly.getBounds(), { padding: [60, 60] });
    } else if (latlngs.length === 1) {
      map.setView(latlngs[0], 14);
    }

    // Numbered Markers
    points.forEach((pt, index) => {
      const isFirst = index === 0;
      const isLast = index === points.length - 1;
      let pinClass = 'custom-pin';
      if (isFirst) pinClass += ' start';
      if (isLast && !isFirst) pinClass += ' end';

      const customIcon = L.divIcon({
        className: 'custom-pin-wrapper',
        html: `<div class="${pinClass}">${index + 1}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
        popupAnchor: [0, -18]
      });

      const marker = L.marker([pt.latitude, pt.longitude], { icon: customIcon });

      const popupHtml = `
        <div style="font-family:Inter,sans-serif; min-width:200px; color:#fff;">
          <div style="font-size:13px; font-weight:700; color:#38bdf8; margin-bottom:6px;">
            #${index + 1} ${pt.camera_id}
          </div>
          <div style="font-size:11px; margin:3px 0; color:#94a3b8;">Plate: <strong>${pt.cleaned_plate}</strong></div>
          <div style="font-size:11px; margin:3px 0; color:#94a3b8;">Time: <strong>${new Date(pt.timestamp).toLocaleTimeString()}</strong></div>
          <div style="font-size:11px; margin:3px 0; color:#94a3b8;">Coords: <strong>${pt.latitude.toFixed(4)}, ${pt.longitude.toFixed(4)}</strong></div>
          <div style="font-size:11px; margin:3px 0; color:#94a3b8;">Leg Dist: <strong>${pt.distance_from_prev_m.toFixed(1)} m</strong></div>
          <div style="font-size:11px; margin:3px 0; color:#94a3b8;">Speed: <strong>${pt.speed_kmh.toFixed(1)} km/h</strong></div>
        </div>
      `;

      marker.bindPopup(popupHtml);
      markersLayer.current.addLayer(marker);
    });
  };

  return (
    <div>
      <div className="glass-panel" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-muted)' }}>Target Plate:</span>
            <input
              type="text"
              className="form-control"
              style={{ width: '180px', fontFamily: 'JetBrains Mono', fontWeight: 'bold' }}
              value={currentPlate}
              onChange={(e) => setCurrentPlate(e.target.value.toUpperCase())}
            />
            <button className="btn" onClick={() => loadTrajectory(currentPlate)}>
              Reconstruct Trajectory
            </button>
          </div>

          <div style={{ display: 'flex', gap: '6px' }}>
            {['HR51BC3493', 'UP21X3666', 'DL3CC8387', 'ZCL3285'].map((p) => (
              <button
                key={p}
                className="btn"
                style={{ padding: '4px 8px', fontSize: '11px', background: 'rgba(255,255,255,0.06)' }}
                onClick={() => setCurrentPlate(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {trajData && (
        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-card-header"><span>Matched Vehicle</span></div>
            <div className="metric-card-value mono">{trajData.matched_plate}</div>
            <div className="metric-card-sub">{trajData.match_type}</div>
          </div>
          <div className="metric-card green">
            <div className="metric-card-header"><span>Total Route Length</span></div>
            <div className="metric-card-value">{(trajData.total_distance_meters / 1000).toFixed(2)} km</div>
            <div className="metric-card-sub">{trajData.total_distance_meters.toFixed(0)} meters</div>
          </div>
          <div className="metric-card purple">
            <div className="metric-card-header"><span>Surveillance Nodes</span></div>
            <div className="metric-card-value">{trajData.points.length} Stations</div>
            <div className="metric-card-sub">In chronological sequence</div>
          </div>
          <div className="metric-card amber">
            <div className="metric-card-header"><span>Transit Span</span></div>
            <div className="metric-card-value" style={{ fontSize: '16px' }}>
              {new Date(trajData.first_seen).toLocaleTimeString()} - {new Date(trajData.last_seen).toLocaleTimeString()}
            </div>
            <div className="metric-card-sub">Duration ~ 1 hour 0 min</div>
          </div>
        </div>
      )}

      <div className="map-view-container">
        <div id="leaflet-map"></div>
      </div>
    </div>
  );
}

// =========================================================================
// Component: Traffic Analytics View
// =========================================================================
function TrafficAnalyticsView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAnalytics() {
      try {
        const res = await api.getAnalyticsSummary();
        setData(res);
      } catch (err) {
        console.error('Analytics fetch error:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="loading-box">
        <div className="spinner"></div>
        <p>Computing Traffic Statistics & Spatial Corridor Volumes...</p>
      </div>
    );
  }

  const cameras = data?.cameras || [];
  const hourly = data?.hourly_traffic || [];
  const sp = data?.speed_and_distance || {};
  const routes = data?.common_routes || [];

  return (
    <div>
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-card-header"><span>Average Travel Speed</span></div>
          <div className="metric-card-value">{sp.average_speed_kmh} <span style={{ fontSize: '14px' }}>km/h</span></div>
          <div className="metric-card-sub">Across all 12 inter-camera legs</div>
        </div>
        <div className="metric-card rose">
          <div className="metric-card-header"><span>Max Observed Speed</span></div>
          <div className="metric-card-value">{sp.max_speed_kmh} <span style={{ fontSize: '14px' }}>km/h</span></div>
          <div className="metric-card-sub">Cam 1 to Cam 2 corridor</div>
        </div>
        <div className="metric-card green">
          <div className="metric-card-header"><span>Min Transit Speed</span></div>
          <div className="metric-card-value">{sp.min_speed_kmh} <span style={{ fontSize: '14px' }}>km/h</span></div>
          <div className="metric-card-sub">Ring Road to South Delhi</div>
        </div>
        <div className="metric-card purple">
          <div className="metric-card-header"><span>Avg Path Distance</span></div>
          <div className="metric-card-value">{sp.average_path_distance_km} <span style={{ fontSize: '14px' }}>km</span></div>
          <div className="metric-card-sub">{sp.vehicles_with_multi_point_path} tracked vehicles</div>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel">
          <div className="panel-header">
            <div className="panel-title"><span>📊</span> Camera Surveillance Traffic Density</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {cameras.map((c) => (
              <div key={c.camera_id}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span className="mono">{c.camera_id}</span>
                  <strong>{c.detection_count} detections ({c.vehicle_count} vehicles)</strong>
                </div>
                <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${(c.detection_count / 36) * 100}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #38bdf8, #0284c7)',
                      borderRadius: '4px'
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel">
          <div className="panel-header">
            <div className="panel-title"><span>⏰</span> Hourly Vehicle Flow</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {hourly.map((h) => (
              <div key={h.hour_of_day}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>{h.hour_label}</span>
                  <strong>{h.detection_count} passes</strong>
                </div>
                <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${(h.detection_count / 44) * 100}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #10b981, #059669)',
                      borderRadius: '4px'
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="glass-panel">
        <div className="panel-header">
          <div className="panel-title"><span>🛣️</span> Primary Multi-Node Vehicle Corridors</div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Corridor Sequence</th>
              <th>Volume Frequency</th>
              <th>Sample Registered Vehicles</th>
            </tr>
          </thead>
          <tbody>
            {routes.map((r, idx) => (
              <tr key={idx}>
                <td className="mono" style={{ color: 'var(--accent-blue)' }}>{r.route}</td>
                <td><span className="badge green">{r.frequency} vehicles</span></td>
                <td className="mono" style={{ fontSize: '11px' }}>{r.vehicles.join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =========================================================================
// Component: Alerts & Violations View
// =========================================================================
function AlertsView({ onTrackPlate }) {
  const [alerts, setAlerts] = useState([]);
  const [filterSeverity, setFilterSeverity] = useState('ALL');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadAlerts() {
      try {
        const data = await api.getAlerts();
        setAlerts(data);
      } catch (err) {
        console.error('Alerts load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadAlerts();
  }, []);

  const filtered = alerts.filter((a) => {
    if (filterSeverity === 'ALL') return true;
    return a.severity === filterSeverity || a.alert_type === filterSeverity;
  });

  return (
    <div>
      <div className="glass-panel" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div className="panel-title"><span>🚨</span> Active Law Enforcement & Rule Alerts ({filtered.length})</div>
          
          <div style={{ display: 'flex', gap: '6px' }}>
            {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'WATCHLIST_HIT', 'SPEEDING_VIOLATION'].map((sev) => (
              <button
                key={sev}
                className="btn"
                style={{
                  padding: '4px 10px',
                  fontSize: '11px',
                  background: filterSeverity === sev ? 'var(--accent-blue)' : 'rgba(255,255,255,0.06)'
                }}
                onClick={() => setFilterSeverity(sev)}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="glass-panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>Alert ID</th>
              <th>Vehicle Plate</th>
              <th>Alert Type</th>
              <th>Severity</th>
              <th>Surveillance Node</th>
              <th>Timestamp</th>
              <th>Incident Details</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((a) => (
              <tr key={a.alert_id}>
                <td className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{a.alert_id}</td>
                <td className="mono" style={{ fontWeight: 'bold', color: '#fff' }}>{a.plate}</td>
                <td>
                  <span className="badge" style={{ background: 'rgba(255,255,255,0.08)' }}>
                    {a.alert_type}
                  </span>
                </td>
                <td>
                  <span className={`badge ${a.severity.toLowerCase()}`}>
                    {a.severity}
                  </span>
                </td>
                <td className="mono" style={{ fontSize: '11px' }}>{a.camera_id}</td>
                <td style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{new Date(a.timestamp).toLocaleTimeString()}</td>
                <td style={{ fontSize: '12px', maxWidth: '360px' }}>{a.message}</td>
                <td>
                  <button
                    className="btn"
                    style={{ padding: '3px 8px', fontSize: '10px' }}
                    onClick={() => onTrackPlate(a.plate)}
                  >
                    Track
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =========================================================================
// Component: Watchlist Management View
// =========================================================================
function WatchlistView({ onTrackPlate }) {
  const [watchlist, setWatchlist] = useState([]);
  const [plateInput, setPlateInput] = useState('');
  const [descInput, setDescInput] = useState('');
  const [priorityInput, setPriorityInput] = useState('HIGH');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWatchlist();
  }, []);

  const loadWatchlist = async () => {
    try {
      const data = await api.getWatchlist();
      setWatchlist(data);
    } catch (err) {
      console.error('Watchlist fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!plateInput.trim()) return;

    try {
      await api.addToWatchlist({
        cleaned_plate: plateInput.trim().toUpperCase(),
        description: descInput || 'Target Vehicle for Interception',
        priority: priorityInput,
        enabled: true
      });
      setPlateInput('');
      setDescInput('');
      loadWatchlist();
    } catch (err) {
      alert('Error adding to watchlist: ' + err.message);
    }
  };

  const handleDelete = async (plate) => {
    if (!confirm(`Remove vehicle ${plate} from active watchlist?`)) return;
    try {
      await api.deleteFromWatchlist(plate);
      loadWatchlist();
    } catch (err) {
      alert('Error deleting: ' + err.message);
    }
  };

  return (
    <div>
      <div className="glass-panel" style={{ marginBottom: '20px' }}>
        <div className="panel-header">
          <div className="panel-title"><span>🎯</span> Add Vehicle to Real-Time Watchlist</div>
        </div>

        <form onSubmit={handleAdd} style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ marginBottom: 0, width: '200px' }}>
            <label>Vehicle Plate</label>
            <input
              type="text"
              className="form-control mono"
              value={plateInput}
              placeholder="E.G. HR51BC3493"
              onChange={(e) => setPlateInput(e.target.value.toUpperCase())}
              required
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: '240px' }}>
            <label>Case Reason / Tag</label>
            <input
              type="text"
              className="form-control"
              value={descInput}
              placeholder="E.g. Stolen Vehicle, Suspect Convoy, VIP..."
              onChange={(e) => setDescInput(e.target.value)}
              required
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0, width: '140px' }}>
            <label>Priority</label>
            <select
              className="form-control"
              value={priorityInput}
              onChange={(e) => setPriorityInput(e.target.value)}
            >
              <option value="CRITICAL">CRITICAL</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="LOW">LOW</option>
            </select>
          </div>

          <button type="submit" className="btn" style={{ height: '36px' }}>
            ➕ Add Watchlist Entry
          </button>
        </form>
      </div>

      <div className="glass-panel">
        <div className="panel-header">
          <div className="panel-title"><span>📋</span> Active Target Vehicle Watchlist ({watchlist.length})</div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Plate Number</th>
              <th>Description / Reason</th>
              <th>Priority</th>
              <th>Status</th>
              <th>Created At</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {watchlist.map((item) => (
              <tr key={item.cleaned_plate}>
                <td className="mono" style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--accent-blue)' }}>
                  {item.cleaned_plate}
                </td>
                <td>{item.description}</td>
                <td>
                  <span className={`badge ${item.priority.toLowerCase()}`}>
                    {item.priority}
                  </span>
                </td>
                <td>
                  <span className="badge green">ACTIVE</span>
                </td>
                <td style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  {new Date(item.created_at).toLocaleString()}
                </td>
                <td style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="btn"
                    style={{ padding: '3px 8px', fontSize: '11px' }}
                    onClick={() => onTrackPlate(item.cleaned_plate)}
                  >
                    Track
                  </button>
                  <button
                    className="btn danger"
                    style={{ padding: '3px 8px', fontSize: '11px' }}
                    onClick={() => handleDelete(item.cleaned_plate)}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =========================================================================
// Main App Component with Tab Navigation
// =========================================================================
export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPlate, setSelectedPlate] = useState('HR51BC3493');
  const [health, setHealth] = useState(null);
  const [alertCount, setAlertCount] = useState(0);

  useEffect(() => {
    async function init() {
      try {
        const [hData, aData] = await Promise.all([
          api.getHealth(),
          api.getAlerts()
        ]);
        setHealth(hData);
        setAlertCount(aData?.length || 0);
      } catch (err) {
        console.error('App init error:', err);
      }
    }
    init();
  }, []);

  const handleSelectPlate = (plate) => {
    setSelectedPlate(plate);
    setActiveTab('vehicles');
  };

  const handleTrackOnMap = (plate) => {
    setSelectedPlate(plate);
    setActiveTab('trajectory');
  };

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        alertCount={alertCount}
      />

      <div className="main-wrapper">
        <TopHeader
          activeTab={activeTab}
          onSearchPlate={handleTrackOnMap}
          health={health}
        />

        <div className="content-viewport">
          {activeTab === 'overview' && (
            <OverviewView
              onSelectPlate={handleTrackOnMap}
              onSwitchTab={setActiveTab}
            />
          )}

          {activeTab === 'vehicles' && (
            <VehicleSearchView
              selectedPlate={selectedPlate}
              onSelectPlate={handleSelectPlate}
              onTrackOnMap={handleTrackOnMap}
            />
          )}

          {activeTab === 'trajectory' && (
            <TrajectoryMapView
              targetPlate={selectedPlate}
              onSelectPlate={setSelectedPlate}
            />
          )}

          {activeTab === 'analytics' && (
            <TrafficAnalyticsView />
          )}

          {activeTab === 'alerts' && (
            <AlertsView
              onTrackPlate={handleTrackOnMap}
            />
          )}

          {activeTab === 'watchlist' && (
            <WatchlistView
              onTrackPlate={handleTrackOnMap}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Render into DOM
const rootEl = document.getElementById('root');
if (rootEl) {
  const root = ReactDOM.createRoot(rootEl);
  root.render(<App />);
}
