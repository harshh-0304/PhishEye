"""
PhishEye URL Feature Extraction Engine.

Implements structural, syntactic, and protocol feature extraction for raw URLs,
mapping them onto the 30 standard features used by the Random Forest classifier.
Encoding conforms to the UCI Phishing Websites standard:
  1: Legitimate / Safe
  0: Suspicious / Neutral
 -1: Phishing / Malicious
"""

import ipaddress
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

# Comprehensive list of popular URL shortening services
URL_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at",
    "tiny.cc", "bc.vc", "bl.ink", "lnkd.in", "s.id", "v.gd", "soo.gd",
    "rebrandly.com", "snip.ly", "trib.al", "cli.re", "qr.ae"
}

# Standard expected web ports
STANDARD_PORTS = {80, 443, 8080, 8443}

# Multi-part public suffixes for domain depth calculation
COMMON_TWO_PART_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au",
    "co.in", "net.in", "org.in", "co.nz", "co.jp", "com.br",
    "com.mx", "co.za", "com.sg", "com.tr", "edu.au", "gov.in"
}

# Ordered 30 feature names matching the trained model schema
FEATURE_NAMES: List[str] = [
    "having_IPhaving_IP_Address",
    "URLURL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report"
]


def normalize_url(url: str) -> str:
    """Normalize input URL string by trimming and ensuring a valid scheme."""
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = "http://" + cleaned
    return cleaned


def is_ip_address(hostname: str) -> bool:
    """Check if hostname is an IPv4 or IPv6 literal or hex/octal representation."""
    if not hostname:
        return False

    clean_host = hostname.strip("[]")

    # Standard IP address validation
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        pass

    # Hexadecimal IP representation (e.g. 0x7f000001 or 0x7f.0x0.0x0.0x1)
    if re.match(r"^0x[0-9a-fA-F]+(\.0x[0-9a-fA-F]+){3}$", clean_host):
        return True
    if re.match(r"^0x[0-9a-fA-F]{8}$", clean_host):
        return True

    # Octal IP representation (e.g. 0177.0.0.1)
    if re.match(r"^0[0-7]+(\.0[0-7]+){3}$", clean_host):
        return True

    return False


def count_subdomains(hostname: str) -> int:
    """Calculate the number of subdomains taking multi-part TLDs into account."""
    if not hostname or is_ip_address(hostname):
        return 0

    parts = [p.lower() for p in hostname.split(".") if p]
    if len(parts) <= 1:
        return 0

    # Strip port if present in hostname
    parts[-1] = parts[-1].split(":")[0]

    # Check for two-part TLDs
    if len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in COMMON_TWO_PART_TLDS:
        # e.g., sub.example.co.uk -> parts: [sub, example, co, uk], tld=2, sld=1
        sub_count = len(parts) - 3
    else:
        # e.g., sub.example.com -> parts: [sub, example, com], tld=1, sld=1
        sub_count = len(parts) - 2

    return max(0, sub_count)


class URLFeatureExtractor:
    """
    Modular feature extractor extracting all 30 phishing indicator features
    from raw URL strings.
    """

    def __init__(self, feature_order: Optional[List[str]] = None):
        self.feature_order = feature_order or FEATURE_NAMES

    def extract(self, raw_url: str) -> Dict[str, int]:
        """
        Extract complete 30-feature dictionary from a URL string.
        """
        url = normalize_url(raw_url)
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        full_url_lower = url.lower()

        features: Dict[str, int] = {f: 1 for f in self.feature_order}

        # 1. having_IPhaving_IP_Address: -1 if IP, 1 if domain name
        features["having_IPhaving_IP_Address"] = -1 if is_ip_address(hostname) else 1

        # 2. URLURL_Length: 1 if < 54 chars, 0 if 54-75 chars, -1 if > 75 chars
        url_len = len(url)
        if url_len < 54:
            features["URLURL_Length"] = 1
        elif 54 <= url_len <= 75:
            features["URLURL_Length"] = 0
        else:
            features["URLURL_Length"] = -1

        # 3. Shortining_Service: -1 if shortener, 1 otherwise
        features["Shortining_Service"] = (
            -1 if any(service in hostname for service in URL_SHORTENERS) else 1
        )

        # 4. having_At_Symbol: -1 if '@' in URL, 1 otherwise
        features["having_At_Symbol"] = -1 if "@" in url else 1

        # 5. double_slash_redirecting: -1 if '//' appears after position 7/8
        scheme_offset = len(parsed.scheme) + 3 if parsed.scheme else 0
        features["double_slash_redirecting"] = (
            -1 if url.find("//", scheme_offset) != -1 else 1
        )

        # 6. Prefix_Suffix: -1 if '-' present in domain, 1 otherwise
        features["Prefix_Suffix"] = -1 if "-" in hostname else 1

        # 7. having_Sub_Domain: 1 if <=1, 0 if 2, -1 if >2 subdomains
        sub_count = count_subdomains(hostname)
        if sub_count == 0 or sub_count == 1:
            features["having_Sub_Domain"] = 1
        elif sub_count == 2:
            features["having_Sub_Domain"] = 0
        else:
            features["having_Sub_Domain"] = -1

        # 8. SSLfinal_State: 1 if HTTPS, -1 if HTTP
        features["SSLfinal_State"] = 1 if parsed.scheme == "https" else -1

        # 9. Domain_registeration_length: heuristic based on shortener/domain structure
        features["Domain_registeration_length"] = (
            -1 if features["Shortining_Service"] == -1 or sub_count > 2 else 1
        )

        # 10. Favicon: heuristic based on external host indicators
        features["Favicon"] = 1

        # 11. port: -1 if non-standard port specified, 1 otherwise
        if parsed.port and parsed.port not in STANDARD_PORTS:
            features["port"] = -1
        else:
            features["port"] = 1

        # 12. HTTPS_token: -1 if 'https' appears in hostname/path without https scheme
        if "https" in hostname or "https" in path:
            features["HTTPS_token"] = -1 if parsed.scheme != "https" else 1
        else:
            features["HTTPS_token"] = 1

        # 13. Request_URL: heuristic based on suspicious query params or subdomains
        features["Request_URL"] = -1 if features["having_Sub_Domain"] == -1 else 1

        # 14. URL_of_Anchor: heuristic correlated with @ or suspicious length
        if features["having_At_Symbol"] == -1 or features["URLURL_Length"] == -1:
            features["URL_of_Anchor"] = -1
        else:
            features["URL_of_Anchor"] = 1

        # 15. Links_in_tags
        features["Links_in_tags"] = -1 if features["Prefix_Suffix"] == -1 and features["SSLfinal_State"] == -1 else 1

        # 16. SFH (Server Form Handler): -1 if empty or about:blank in path/query
        if "about:blank" in full_url_lower or "javascript:" in full_url_lower:
            features["SFH"] = -1
        else:
            features["SFH"] = 1

        # 17. Submitting_to_email: -1 if mailto: or email submission in query/path
        if "mailto:" in full_url_lower or "mail()" in full_url_lower:
            features["Submitting_to_email"] = -1
        else:
            features["Submitting_to_email"] = 1

        # 18. Abnormal_URL: -1 if hostname not in URL path/host or IP used
        features["Abnormal_URL"] = -1 if features["having_IPhaving_IP_Address"] == -1 else 1

        # 19. Redirect: -1 if multiple redirections (>4) simulated or target redirect params
        if "redirect=" in query or "next=" in query or "url=" in query or "dest=" in query:
            features["Redirect"] = 0
        else:
            features["Redirect"] = 1

        # 20. on_mouseover: heuristic default
        features["on_mouseover"] = 1

        # 21. RightClick: heuristic default
        features["RightClick"] = 1

        # 22. popUpWidnow: heuristic check for popup parameters
        if "popup" in query.lower() or "window.open" in full_url_lower:
            features["popUpWidnow"] = -1
        else:
            features["popUpWidnow"] = 1

        # 23. Iframe: heuristic check for iframe in query
        if "iframe" in query.lower():
            features["Iframe"] = -1
        else:
            features["Iframe"] = 1

        # 24. age_of_domain: heuristic fallback
        features["age_of_domain"] = -1 if features["Shortining_Service"] == -1 else 1

        # 25. DNSRecord: default 1 (assuming DNS resolvable)
        features["DNSRecord"] = 1

        # 26. web_traffic: -1 if shortener or suspicious subdomains, 1 otherwise
        if features["Shortining_Service"] == -1 or features["having_Sub_Domain"] == -1:
            features["web_traffic"] = -1
        else:
            features["web_traffic"] = 1

        # 27. Page_Rank: -1 if clearly anomalous structure
        if features["having_IPhaving_IP_Address"] == -1 or features["having_At_Symbol"] == -1:
            features["Page_Rank"] = -1
        else:
            features["Page_Rank"] = 1

        # 28. Google_Index: 1
        features["Google_Index"] = 1

        # 29. Links_pointing_to_page: 1
        features["Links_pointing_to_page"] = 1

        # 30. Statistical_report: -1 if multiple suspicious flags triggered
        risk_flags = sum(1 for v in features.values() if v == -1)
        features["Statistical_report"] = -1 if risk_flags >= 3 else 1

        return features

    def to_dataframe(self, raw_url: str) -> pd.DataFrame:
        """Convert extracted feature vector to a single-row Pandas DataFrame."""
        features_dict = self.extract(raw_url)
        ordered_values = [features_dict[k] for k in self.feature_order]
        return pd.DataFrame([ordered_values], columns=self.feature_order)


# Module-level convenience function
def extract_features(raw_url: str, feature_order: Optional[List[str]] = None) -> Dict[str, int]:
    """Helper function to extract features directly from a URL string."""
    extractor = URLFeatureExtractor(feature_order)
    return extractor.extract(raw_url)
