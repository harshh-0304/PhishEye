"""Unit tests for the PhishEye brand typosquatting analyzer."""

import unittest
from phisheye.typosquat import (
    levenshtein_distance,
    check_typosquatting,
    normalize_substitutions,
    extract_base_labels
)


class TestTyposquatting(unittest.TestCase):

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("paypal", "paypal"), 0)
        self.assertEqual(levenshtein_distance("paypal", "paypa1"), 1)
        self.assertEqual(levenshtein_distance("google", "g00gle"), 2)
        self.assertEqual(levenshtein_distance("", "test"), 4)

    def test_normalize_substitutions(self):
        self.assertEqual(normalize_substitutions("paypa1"), "paypal")
        self.assertEqual(normalize_substitutions("g00gle"), "google")
        self.assertEqual(normalize_substitutions("vvord"), "word")

    def test_official_brand_domain(self):
        res = check_typosquatting("https://www.paypal.com")
        self.assertFalse(res.is_impersonating)
        self.assertEqual(res.risk_level, "SAFE")

    def test_combosquatting_detection(self):
        res = check_typosquatting("https://paypal-update-account.com")
        self.assertTrue(res.is_impersonating)
        self.assertEqual(res.impersonated_brand, "paypal")
        self.assertEqual(res.attack_vector, "COMBOSQUATTING")

    def test_typosquatting_detection(self):
        res = check_typosquatting("http://paypa1.com")
        self.assertTrue(res.is_impersonating)
        self.assertEqual(res.impersonated_brand, "paypal")
        self.assertEqual(res.attack_vector, "TYPOSQUATTING")

    def test_unrelated_domain(self):
        res = check_typosquatting("https://stackoverflow.com")
        self.assertFalse(res.is_impersonating)
        self.assertEqual(res.risk_level, "CLEAN")


if __name__ == "__main__":
    unittest.main()
