"""Unit tests for PhishEye REST API endpoints."""

from http.server import HTTPServer
import json
import threading
import time
import unittest
import urllib.request

from phisheye.api import PhishEyeAPIHandler


class TestPhishEyeAPI(unittest.TestCase):
    server: HTTPServer
    server_thread: threading.Thread
    port: int = 18099
    base_url: str

    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server = HTTPServer(("127.0.0.1", cls.port), PhishEyeAPIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/v1/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["model_loaded"])

    def test_features_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/v1/features")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["feature_count"], 30)

    def test_scan_endpoint_safe(self):
        payload = json.dumps({"url": "https://google.com"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/scan",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertFalse(data["is_phishing"])
            self.assertEqual(data["verdict"], "Safe Domain")

    def test_scan_endpoint_phishing(self):
        payload = json.dumps({"url": "http://paypal-security-account.login-auth.com/verify@user"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/scan",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["is_phishing"])
            self.assertEqual(data["verdict"], "Phishing Detected")
            self.assertIn("homograph", data)
            self.assertIn("entropy", data)


if __name__ == "__main__":
    unittest.main()
