"""
TraceX SIH 2026 - GIS Trajectory Web Server
Lightweight Python HTTP server serving the Leaflet GIS Map and Trajectory Reconstruction API.
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trajectory.trajectory_service import TrajectoryService

service = TrajectoryService()

class TrajectoryHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, content_type="text/html"):
        if not os.path.exists(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # 1. API: /api/trajectory?plate=HR51BC3493&fuzzy=true
        if path == "/api/trajectory":
            plate = params.get("plate", [""])[0].strip()
            allow_fuzzy = params.get("fuzzy", ["true"])[0].lower() in ("true", "1", "yes")
            max_dist = int(params.get("max_dist", ["2"])[0])

            if not plate:
                self._send_json({"error": "Plate parameter is required"}, status=400)
                return

            try:
                trajectories = service.query_trajectory(
                    plate=plate,
                    allow_fuzzy=allow_fuzzy,
                    max_edit_distance=max_dist
                )
                self._send_json({
                    "searched_plate": plate,
                    "count": len(trajectories),
                    "trajectories": [t.to_dict() for t in trajectories]
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 2. API: /api/plates
        elif path == "/api/plates":
            try:
                unique_plates = service.get_all_unique_plates()
                self._send_json({
                    "count": len(unique_plates),
                    "plates": unique_plates
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 3. Static Web Pages: / or /index.html
        elif path in ("/", "/index.html"):
            index_path = os.path.join(os.path.dirname(__file__), "index.html")
            self._send_file(index_path, "text/html")
            return

        # 4. Fallback 404
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

def run_server(port=8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, TrajectoryHTTPHandler)
    print(f"TraceX GIS Server running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
