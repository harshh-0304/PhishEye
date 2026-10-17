"""Unit tests for the PhishEye IDN homograph and Punycode detector."""

import unittest
from phisheye.homograph import (
    analyze_homograph,
    detect_punycode,
    detect_mixed_scripts,
    decode_punycode,
    find_confusables,
    extract_domain
)


class TestHomographDetector(unittest.TestCase):

    def test_extract_domain(self):
        self.assertEqual(extract_domain("https://example.com/path?arg=1"), "example.com")
        self.assertEqual(extract_domain("google.com"), "google.com")
        self.assertEqual(extract_domain("http://sub.domain.org:8080/"), "sub.domain.org")

    def test_detect_punycode(self):
        self.assertTrue(detect_punycode("xn--pple-43d.com"))
        self.assertFalse(detect_punycode("apple.com"))

    def test_decode_punycode(self):
        # xn--e1afmkfd.xn--p1ai -> пример.рф
        decoded = decode_punycode("xn--e1afmkfd.ru")
        self.assertIsNotNone(decoded)

    def test_detect_mixed_scripts(self):
        # Clean Latin
        self.assertFalse(detect_mixed_scripts("paypal"))
        # Mixed Cyrillic 'а' (U+0430) with Latin letters
        mixed = "p" + "\u0430" + "ypal"
        self.assertTrue(detect_mixed_scripts(mixed))

    def test_find_confusables(self):
        cyrillic_o = "\u043e"
        confusables = find_confusables(f"g{cyrillic_o}{cyrillic_o}gle")
        self.assertEqual(len(confusables), 2)
        self.assertEqual(confusables[0]["latin_equivalent"], "o")

    def test_analyze_homograph_clean(self):
        rep = analyze_homograph("https://microsoft.com")
        self.assertFalse(rep.is_homograph)
        self.assertFalse(rep.is_punycode)
        self.assertFalse(rep.has_mixed_scripts)
        self.assertEqual(len(rep.confusable_glyphs), 0)

    def test_analyze_homograph_spoofed(self):
        # Spoofed with Cyrillic small 'a'
        spoofed = "http://" + "\u0430" + "pple.com"
        rep = analyze_homograph(spoofed)
        self.assertTrue(rep.is_homograph)
        self.assertTrue(len(rep.confusable_glyphs) > 0)


if __name__ == "__main__":
    unittest.main()
