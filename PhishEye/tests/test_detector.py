"""Integration tests for PhishEyeDetector model inference pipeline."""

import unittest
from phisheye.detector import PhishEyeDetector, ScanResult


class TestPhishEyeDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = PhishEyeDetector()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(len(self.detector.feature_names), 30)

    def test_predict_safe_url(self):
        result = self.detector.predict("https://google.com")
        self.assertIsInstance(result, ScanResult)
        self.assertFalse(result.is_phishing)
        self.assertEqual(result.verdict, "Safe Domain")
        self.assertGreaterEqual(result.confidence, 50.0)
        self.assertIn("safe", result.probabilities)
        self.assertIn("phishing", result.probabilities)

    def test_predict_phishing_url(self):
        suspicious_url = "http://paypal-security-account.login-auth.com/verify@user"
        result = self.detector.predict(suspicious_url)
        self.assertIsInstance(result, ScanResult)
        self.assertTrue(result.is_phishing)
        self.assertEqual(result.verdict, "Phishing Detected")
        self.assertGreater(len(result.risk_indicators), 0)

    def test_scan_result_to_dict(self):
        result = self.detector.predict("https://example.com")
        d = result.to_dict()
        self.assertIn("url", d)
        self.assertIn("verdict", d)
        self.assertIn("is_phishing", d)
        self.assertIn("confidence", d)
        self.assertIn("risk_score", d)
        self.assertIn("features", d)
        self.assertEqual(len(d["features"]), 30)

    def test_predict_batch(self):
        urls = [
            "https://google.com",
            "https://github.com",
            "http://phishing-scam-bank.com/login@test"
        ]
        results = self.detector.predict_batch(urls)
        self.assertEqual(len(results), 3)
        self.assertFalse(results[0].is_phishing)
        self.assertFalse(results[1].is_phishing)
        self.assertTrue(results[2].is_phishing)


if __name__ == "__main__":
    unittest.main()
