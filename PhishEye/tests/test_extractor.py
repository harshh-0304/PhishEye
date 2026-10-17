"""Unit tests for the PhishEye URLFeatureExtractor module."""

import unittest
from phisheye.extractor import (
    URLFeatureExtractor,
    normalize_url,
    is_ip_address,
    count_subdomains,
    FEATURE_NAMES
)


class TestURLFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = URLFeatureExtractor()

    def test_normalize_url(self):
        self.assertEqual(normalize_url("google.com"), "http://google.com")
        self.assertEqual(normalize_url("https://secure.bank.com"), "https://secure.bank.com")
        self.assertEqual(normalize_url(""), "")

    def test_is_ip_address(self):
        # Valid standard IPs
        self.assertTrue(is_ip_address("192.168.1.1"))
        self.assertTrue(is_ip_address("10.0.0.1"))
        self.assertTrue(is_ip_address("::1"))
        # Hex / Octal representations
        self.assertTrue(is_ip_address("0x7f.0x0.0x0.0x1"))
        # Standard hostnames
        self.assertFalse(is_ip_address("google.com"))
        self.assertFalse(is_ip_address("paypal.secure-login.org"))
        self.assertFalse(is_ip_address(""))

    def test_count_subdomains(self):
        self.assertEqual(count_subdomains("example.com"), 0)
        self.assertEqual(count_subdomains("sub.example.com"), 1)
        self.assertEqual(count_subdomains("a.b.example.com"), 2)
        # Multi-part TLDs
        self.assertEqual(count_subdomains("example.co.uk"), 0)
        self.assertEqual(count_subdomains("sub.example.co.uk"), 1)

    def test_feature_count_and_keys(self):
        features = self.extractor.extract("https://example.com")
        self.assertEqual(len(features), 30)
        for key in FEATURE_NAMES:
            self.assertIn(key, features)

    def test_to_dataframe(self):
        df = self.extractor.to_dataframe("https://example.com")
        self.assertEqual(df.shape, (1, 30))
        self.assertListEqual(list(df.columns), FEATURE_NAMES)

    def test_at_symbol_detection(self):
        clean = self.extractor.extract("https://example.com/login")
        injected = self.extractor.extract("https://example.com/verify@user")
        self.assertEqual(clean["having_At_Symbol"], 1)
        self.assertEqual(injected["having_At_Symbol"], -1)

    def test_ssl_scheme_detection(self):
        https_site = self.extractor.extract("https://example.com")
        http_site = self.extractor.extract("http://example.com")
        self.assertEqual(https_site["SSLfinal_State"], 1)
        self.assertEqual(http_site["SSLfinal_State"], -1)

    def test_prefix_suffix_hyphen_detection(self):
        normal = self.extractor.extract("https://example.com")
        hyphenated = self.extractor.extract("https://paypal-update.com")
        self.assertEqual(normal["Prefix_Suffix"], 1)
        self.assertEqual(hyphenated["Prefix_Suffix"], -1)

    def test_shortener_detection(self):
        short = self.extractor.extract("https://bit.ly/3xyz")
        normal = self.extractor.extract("https://github.com")
        self.assertEqual(short["Shortining_Service"], -1)
        self.assertEqual(normal["Shortining_Service"], 1)


if __name__ == "__main__":
    unittest.main()
