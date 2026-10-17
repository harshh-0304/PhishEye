"""Unit tests for the PhishEye Shannon entropy and DGA domain analyzer."""

import unittest
from phisheye.entropy import (
    calculate_shannon_entropy,
    extract_core_domain,
    analyze_domain_entropy
)


class TestEntropyAnalyzer(unittest.TestCase):

    def test_zero_entropy_for_empty_string(self):
        self.assertEqual(calculate_shannon_entropy(""), 0.0)

    def test_zero_entropy_for_single_repeated_char(self):
        self.assertEqual(calculate_shannon_entropy("aaaaaaa"), 0.0)

    def test_binary_entropy(self):
        # Equal distribution of 2 symbols has entropy exactly 1.0
        self.assertAlmostEqual(calculate_shannon_entropy("abababab"), 1.0)

    def test_extract_core_domain(self):
        self.assertEqual(extract_core_domain("https://auth.company.com/login"), "company")
        self.assertEqual(extract_core_domain("example.com"), "example")

    def test_legitimate_domain_entropy(self):
        rep = analyze_domain_entropy("https://google.com")
        self.assertFalse(rep.is_dga_candidate)
        self.assertIn(rep.entropy_category, ("LOW", "NORMAL"))
        self.assertLess(rep.entropy, 3.5)

    def test_dga_random_domain_entropy(self):
        # Highly random alphanumeric string characteristic of DGAs
        rep = analyze_domain_entropy("https://xk98fqz41m0p8w7r2.com")
        self.assertTrue(rep.is_dga_candidate)
        self.assertGreaterEqual(rep.entropy, 3.6)
        self.assertIn(rep.entropy_category, ("HIGH", "EXTREME"))

    def test_to_dict_structure(self):
        rep = analyze_domain_entropy("https://example.com")
        d = rep.to_dict()
        self.assertIn("entropy", d)
        self.assertIn("is_dga_candidate", d)
        self.assertIn("vowel_consonant_ratio", d)


if __name__ == "__main__":
    unittest.main()
