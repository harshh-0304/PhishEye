"""
PhishEye Shannon Entropy & DGA Domain Randomness Engine.

Calculates the information entropy of domain names to detect algorithmic
randomness indicative of Domain Generation Algorithms (DGA) commonly employed
by command-and-control malware and disposable phishing infrastructure.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional
from urllib.parse import urlparse


@dataclass
class EntropyReport:
    """Detailed entropy analysis of a domain name."""
    domain: str
    target_string: str
    entropy: float
    is_dga_candidate: bool
    entropy_category: str  # LOW, NORMAL, HIGH, EXTREME
    vowel_consonant_ratio: float
    digit_ratio: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "target_string": self.target_string,
            "entropy": round(self.entropy, 4),
            "is_dga_candidate": self.is_dga_candidate,
            "entropy_category": self.entropy_category,
            "vowel_consonant_ratio": round(self.vowel_consonant_ratio, 4),
            "digit_ratio": round(self.digit_ratio, 4),
            "description": self.description,
        }


def calculate_shannon_entropy(text: str) -> float:
    """
    Compute the Shannon information entropy of a given string.
    Formula: H = -sum(p(x) * log2(p(x)))
    """
    if not text:
        return 0.0

    length = len(text)
    counts = Counter(text)
    entropy = 0.0

    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def extract_core_domain(url_or_domain: str) -> str:
    """Extract the primary domain label for entropy calculation."""
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return ""

    # Get the registered domain label without TLD
    parts = [p for p in hostname.split(".") if p]
    if len(parts) >= 2:
        return parts[-2]
    return hostname


def analyze_domain_entropy(url_or_domain: str) -> EntropyReport:
    """
    Evaluate domain randomness and assess risk of DGA (Domain Generation Algorithm).
    """
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower()

    core_label = extract_core_domain(url_or_domain)
    target = core_label if core_label else hostname
    entropy = calculate_shannon_entropy(target)

    # Calculate vowel/consonant and digit ratios
    alpha_chars = [c for c in target if c.isalpha()]
    vowels = sum(1 for c in alpha_chars if c in "aeiou")
    consonants = len(alpha_chars) - vowels
    v_c_ratio = (vowels / consonants) if consonants > 0 else (1.0 if vowels > 0 else 0.0)

    digit_chars = sum(1 for c in target if c.isdigit())
    digit_ratio = (digit_chars / len(target)) if len(target) > 0 else 0.0

    # Classification logic based on character entropy thresholds
    # Normal human-readable brand names: 2.2 - 3.5
    # High randomness (DGA): > 3.8
    if entropy >= 4.0 or (entropy >= 3.6 and digit_ratio > 0.3):
        category = "EXTREME"
        is_dga = True
        desc = "High algorithmic randomness detected; strongly indicative of DGA / disposable phishing host."
    elif entropy >= 3.5:
        category = "HIGH"
        is_dga = (v_c_ratio < 0.2 or digit_ratio > 0.3)
        desc = "Elevated entropy level; domain name exhibits irregular randomness patterns."
    elif entropy >= 2.0:
        category = "NORMAL"
        is_dga = False
        desc = "Standard lexical entropy typical of natural language domain names."
    else:
        category = "LOW"
        is_dga = False
        desc = "Low entropy; repetitive character sequences."

    return EntropyReport(
        domain=hostname,
        target_string=target,
        entropy=entropy,
        is_dga_candidate=is_dga,
        entropy_category=category,
        vowel_consonant_ratio=v_c_ratio,
        digit_ratio=digit_ratio,
        description=desc
    )
