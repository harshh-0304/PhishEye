"""Unit tests for the PhishEye URL defanging and threat sanitization utilities."""

import unittest
from phisheye.security import defang_url, refang_url, is_defanged


class TestSecurityUtilities(unittest.TestCase):

    def test_defang_https_url(self):
        url = "https://phishing-site.example.com/login"
        defanged = defang_url(url)
        self.assertTrue(defanged.startswith("hxxps[://]"))
        self.assertIn("[.]", defanged)
        self.assertNotIn(".", defanged.split("/")[0].replace("[.]", ""))

    def test_defang_http_url(self):
        url = "http://evil.com"
        defanged = defang_url(url)
        self.assertEqual(defanged, "hxxp[://]evil[.]com")

    def test_refang_url(self):
        defanged = "hxxps[://]evil[.]example[.]com/path"
        refanged = refang_url(defanged)
        self.assertEqual(refanged, "https://evil.example.com/path")

    def test_roundtrip_fidelity(self):
        test_urls = [
            "https://login.microsoft.com/auth",
            "http://192.168.1.100/admin",
            "https://test.co.uk/sample?id=99&token=abc",
        ]
        for original in test_urls:
            defanged = defang_url(original)
            restored = refang_url(defanged)
            self.assertEqual(restored, original, f"Roundtrip failed for {original}")

    def test_is_defanged_detection(self):
        self.assertTrue(is_defanged("hxxps[://]evil[.]com"))
        self.assertTrue(is_defanged("http://192[.]168[.]1[.]1"))
        self.assertFalse(is_defanged("https://google.com"))
        self.assertFalse(is_defanged(""))

    def test_empty_string_handling(self):
        self.assertEqual(defang_url(""), "")
        self.assertEqual(refang_url(""), "")
        self.assertFalse(is_defanged(""))


if __name__ == "__main__":
    unittest.main()
