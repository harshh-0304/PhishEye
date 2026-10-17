"""
PhishEye Brand Impersonation & Typosquatting Analysis Engine.

Detects deceptive brand impersonation, edit-distance (Levenshtein) typosquatting,
and visual brand trickery targeting high-profile institutions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# High-profile global brands commonly targeted in phishing campaigns
TARGET_BRANDS: Dict[str, str] = {
    "google": "google.com",
    "paypal": "paypal.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "chase": "chase.com",
    "bankofamerica": "bankofamerica.com",
    "wellsfargo": "wellsfargo.com",
    "binance": "binance.com",
    "coinbase": "coinbase.com",
    "dropbox": "dropbox.com",
    "linkedin": "linkedin.com",
    "twitter": "twitter.com",
    "github": "github.com",
    "steam": "steampowered.com",
    "yahoo": "yahoo.com",
    "outlook": "outlook.com"
}


@dataclass
class TyposquatResult:
    """Findings from brand impersonation and typosquatting inspection."""
    domain: str
    is_impersonating: bool
    impersonated_brand: Optional[str]
    legitimate_domain: Optional[str]
    attack_vector: Optional[str]
    risk_level: str
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "is_impersonating": self.is_impersonating,
            "impersonated_brand": self.impersonated_brand,
            "legitimate_domain": self.legitimate_domain,
            "attack_vector": self.attack_vector,
            "risk_level": self.risk_level,
            "details": self.details,
        }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_base_labels(hostname: str) -> List[str]:
    """Extract individual domain labels excluding public suffixes."""
    parts = [p.lower() for p in hostname.split(".") if p]
    if len(parts) > 1:
        # Exclude common TLDs like com, org, net, co.uk
        return parts[:-1]
    return parts


def normalize_substitutions(text: str) -> str:
    """Normalize common character substitutions (l/1, o/0, rn/m)."""
    s = text.lower()
    s = s.replace("0", "o")
    s = s.replace("1", "l")
    s = s.replace("vv", "w")
    s = s.replace("rn", "m")
    return s


def check_typosquatting(url_or_domain: str) -> TyposquatResult:
    """
    Check if domain is impersonating or typosquatting a known brand.
    """
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return TyposquatResult(
            domain="",
            is_impersonating=False,
            impersonated_brand=None,
            legitimate_domain=None,
            attack_vector=None,
            risk_level="INFO",
            details="Empty or invalid domain."
        )

    labels = extract_base_labels(hostname)

    # 1. Check if domain IS the legitimate domain
    for brand, leg_domain in TARGET_BRANDS.items():
        if hostname == leg_domain or hostname.endswith("." + leg_domain):
            return TyposquatResult(
                domain=hostname,
                is_impersonating=False,
                impersonated_brand=brand,
                legitimate_domain=leg_domain,
                attack_vector="NONE",
                risk_level="SAFE",
                details=f"Domain verified as official {brand.capitalize()} property."
            )

    # 2. Check for brand name embedded in subdomains / combo-squatting
    # e.g., paypal-security.com or verify-apple.com or netflix.evil.com
    for brand, leg_domain in TARGET_BRANDS.items():
        for label in labels:
            if brand in label:
                # Brand embedded in hyphenated name or subdomain
                return TyposquatResult(
                    domain=hostname,
                    is_impersonating=True,
                    impersonated_brand=brand,
                    legitimate_domain=leg_domain,
                    attack_vector="COMBOSQUATTING",
                    risk_level="CRITICAL",
                    details=f"Combosquatting detected: brand '{brand}' used within domain label '{label}' on unauthorized host."
                )

    # 3. Check Levenshtein distance on primary registered domain label
    primary_label = labels[-1] if labels else ""
    norm_label = normalize_substitutions(primary_label)

    for brand, leg_domain in TARGET_BRANDS.items():
        dist = levenshtein_distance(primary_label, brand)
        norm_dist = levenshtein_distance(norm_label, brand)

        if dist in (1, 2) or norm_dist == 0 or (norm_dist == 1 and len(brand) > 4):
            return TyposquatResult(
                domain=hostname,
                is_impersonating=True,
                impersonated_brand=brand,
                legitimate_domain=leg_domain,
                attack_vector="TYPOSQUATTING",
                risk_level="HIGH",
                details=f"Typosquatting detected: '{primary_label}' is dangerously close to '{brand}' (distance: {dist})."
            )

    return TyposquatResult(
        domain=hostname,
        is_impersonating=False,
        impersonated_brand=None,
        legitimate_domain=None,
        attack_vector=None,
        risk_level="CLEAN",
        details="No known brand typosquatting or combo-squatting detected."
    )
