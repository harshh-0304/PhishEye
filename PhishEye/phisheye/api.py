"""
PhishEye Lightweight REST API Engine.

Exposes RESTful endpoints for integrating PhishEye into Security Operations Center
(SOC) workflows, SIEM platforms, and automated security orchestration (SOAR) pipelines.
Built with standard library HTTP server to maximize portable deployment without
imposing external framework dependencies.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .detector import PhishEyeDetector
from .entropy import analyze_domain_entropy
from .homograph import analyze_homograph
from .security import is_defanged, refang_url
from .tld_intel import evaluate_tld_reputation
from .typosquat import check_typosquatting

logger = logging.getLogger("phisheye.api")


class PhishEyeAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler dispatching REST endpoints for threat analysis."""

    detector: Optional[PhishEyeDetector] = None

    @classmethod
    def get_detector(cls) -> PhishEyeDetector:
        if cls.detector is None:
            cls.detector = PhishEyeDetector()
        return cls.detector

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        """Helper to send JSON response with standard security headers."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Handle GET requests for health and schema inspection."""
        parsed_path = urlparse(self.path).path

        if parsed_path in ("/health", "/api/v1/health"):
            detector = self.get_detector()
            self._send_json(200, {
                "status": "healthy",
                "service": "PhishEye Threat Intelligence API",
                "version": "2.1.0",
                "model_loaded": detector.model is not None,
                "feature_count": len(detector.feature_names)
            })
        elif parsed_path in ("/features", "/api/v1/features"):
            detector = self.get_detector()
            self._send_json(200, {
                "feature_count": len(detector.feature_names),
                "features": detector.feature_names
            })
        else:
            self._send_json(404, {"error": "Endpoint not found", "path": parsed_path})

    def do_POST(self):
        """Handle POST requests for single and batch URL analysis."""
        parsed_path = urlparse(self.path).path

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {str(e)}"})
            return

        detector = self.get_detector()

        if parsed_path in ("/scan", "/api/v1/scan"):
            target_url = payload.get("url")
            if not target_url or not isinstance(target_url, str):
                self._send_json(400, {"error": "Missing required 'url' string parameter"})
                return

            active_url = refang_url(target_url) if is_defanged(target_url) else target_url
            scan = detector.predict(active_url)

            response_data = scan.to_dict()
            response_data["homograph"] = analyze_homograph(active_url).to_dict()
            response_data["typosquat"] = check_typosquatting(active_url).to_dict()
            response_data["entropy"] = analyze_domain_entropy(active_url).to_dict()
            response_data["tld"] = evaluate_tld_reputation(active_url).to_dict()

            self._send_json(200, response_data)

        elif parsed_path in ("/batch-scan", "/api/v1/batch-scan"):
            urls = payload.get("urls")
            if not urls or not isinstance(urls, list):
                self._send_json(400, {"error": "Missing required 'urls' list parameter"})
                return

            results = []
            for u in urls[:100]:  # Limit to 100 per request
                if not isinstance(u, str):
                    continue
                active_url = refang_url(u) if is_defanged(u) else u
                scan = detector.predict(active_url)
                item = scan.to_dict()
                results.append(item)

            self._send_json(200, {
                "total": len(results),
                "phishing_count": sum(1 for r in results if r["is_phishing"]),
                "safe_count": sum(1 for r in results if not r["is_phishing"]),
                "results": results
            })
        else:
            self._send_json(404, {"error": "Endpoint not found", "path": parsed_path})

    def log_message(self, format, *args):
        # Override standard logging to reduce terminal clutter in production
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the PhishEye API HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, PhishEyeAPIHandler)
    print(f"PhishEye Threat Intelligence API serving at http://{host}:{port}/api/v1/health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        httpd.server_close()
