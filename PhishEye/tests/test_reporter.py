"""Unit tests for PhishEye report generation utilities."""

import json
import unittest
from phisheye.detector import PhishEyeDetector
from phisheye.reporter import (
    generate_markdown_report,
    generate_json_report,
    generate_csv_report
)


class TestReporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = PhishEyeDetector()
        cls.safe_scan = cls.detector.predict("https://google.com")
        cls.phish_scan = cls.detector.predict("http://paypal-security-account.login-auth.com/verify@user")

    def test_markdown_report(self):
        md = generate_markdown_report(self.phish_scan, analyst_notes="Test investigation.")
        self.assertIn("# 🛡️ PhishEye Threat Intelligence Report", md)
        self.assertIn("PHISHING DETECTED", md)
        self.assertIn("Test investigation.", md)
        self.assertIn("30-Dimension Feature Vector", md)

    def test_json_report(self):
        json_str = generate_json_report(self.safe_scan)
        data = json.loads(json_str)
        self.assertEqual(data["verdict"], "Safe Domain")
        self.assertIn("generated_at", data)
        self.assertIn("defanged_url", data)

    def test_csv_report(self):
        csv_str = generate_csv_report([self.safe_scan, self.phish_scan])
        lines = csv_str.strip().split("\n")
        self.assertEqual(len(lines), 3)  # Header + 2 data rows
        self.assertIn("url,verdict,is_phishing", lines[0])


if __name__ == "__main__":
    unittest.main()
