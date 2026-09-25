"""
TraceX SIH 2026 - Frontend Development Server
Serves the React dashboard on http://localhost:3000 with proper MIME types.
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".jsx"):
            return "text/javascript; charset=utf-8"
        if path.endswith(".js"):
            return "text/javascript; charset=utf-8"
        return super().guess_type(path)

def run_server(port=3000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DashboardHTTPHandler)
    print(f"TraceX React Dashboard running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 3000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
