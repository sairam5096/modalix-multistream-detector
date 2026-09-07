#!/usr/bin/env python3
# Web app on the 132.125 host: serves the viewer + proxies the SOM detector's
# API (detections + model control) so the browser talks to one origin. Video is
# HLS straight from the local MediaMTX (:8888).
import os, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOM = os.environ.get("SOM_API", "http://SET-SOM-IP:8600")
DIR = os.path.dirname(os.path.abspath(__file__))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _proxy(self, method):
        url = SOM + self.path
        data = None
        if method == "POST":
            n = int(self.headers.get("Content-Length", 0) or 0)
            data = self.rfile.read(n)
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read()
                ct = r.headers.get("Content-Type", "application/json")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(502, "SOM API: " + str(e))

    def _file(self):
        path = "/index.html" if self.path in ("/", "") else self.path.split("?")[0]
        fp = os.path.normpath(os.path.join(DIR, path.lstrip("/")))
        if not fp.startswith(DIR) or not os.path.isfile(fp):
            self.send_error(404); return
        body = open(fp, "rb").read()
        ct = "text/html" if fp.endswith(".html") else "application/javascript" if fp.endswith(".js") else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p.startswith("/api/") or p.startswith("/crop/"):
            self._proxy("GET")
        else:
            self._file()

    def do_POST(self):
        self._proxy("POST") if self.path.startswith("/api/") else self.send_error(404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"multimodel-web on :{port}  (proxying API -> {SOM})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
