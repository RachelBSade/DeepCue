"""
Static file server for local frontend development.

Plain `python -m http.server` reads MIME types from the Windows registry,
which often lacks a correct entry for .js files and serves them as
text/plain — browsers then refuse to execute them as ES modules
(type="module" in index.html / interview.html). This script forces the
correct MIME type before starting the server.

Usage:
    cd frontend
    python serve.py
"""
import http.server
import mimetypes
import socketserver

PORT = 5500

# Python 3.12+ guess_type() reads from the mimetypes module directly, not the
# old class-level extensions_map — patch the registry itself so it overrides
# whatever (often wrong) entry Windows has for .js.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

with socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Serving frontend at http://localhost:{PORT}")
    httpd.serve_forever()
