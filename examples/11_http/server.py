import http.server, socketserver, json
class H(http.server.BaseHTTPRequestHandler):
    def _send(self, body):
        b = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("X-Iris-Test", "marker-value")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):  self._send("GET:" + self.path)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        self._send("POST:" + body)
    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        self._send("PUT:" + body)
    def log_message(self, *a): pass
socketserver.TCPServer(("127.0.0.1", 8765), H).serve_forever()
