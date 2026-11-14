"""Static dev server for CamPose (no caching, correct wasm MIME). Usage: python scripts/serve.py [port]"""
import http.server, os, sys

class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
print(f"CamPose: http://localhost:{port}/web/")
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
