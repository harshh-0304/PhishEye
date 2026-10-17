"""
PhishEye TLD Reputation & Shortener Intelligence Engine.

Provides automated threat assessment for Top-Level Domains (TLDs) frequently
abused by threat actors for disposable landing pages, and maintains a registry
of link shorteners.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# TLDs with historically high abuse rates in cyber threat intelligence feeds
HIGH_RISK_TLDS: Dict[str, str] = {
    "zip": "File extension collision risk (often used for payload disguise).",
    "mov": "Media extension collision risk (often used for payload disguise).",
    "top": "Frequently identified in bulk disposable spam & phishing deployments.",
    "xyz": "High abuse volume across automated domain generation infrastructure.",
    "icu": "Commonly exploited for mass SMS phishing (smishing) campaigns.",
    "loan": "High risk associated with predatory financial and fake banking pages.",
    "work": "Commonly leveraged in fake invoice and recruitment phishing scams.",
    "click": "Predominantly used for affiliate fraud, ad redirectors, and phishing.",
    "buzz": "High incidence of automated malware distribution and landing gates.",
    "surf": "Often registered with anonymous credentials for short-lived scams.",
    "tk": "Historically abused free ccTLD with elevated scam activity.",
    "ml": "Historically abused free ccTLD associated with automated bot registration.",
    "ga": "Historically abused free ccTLD commonly used in phishing campaigns.",
    "cf": "Historically abused free ccTLD frequently seen in credential harvesters.",
    "gq": "Historically abused free ccTLD linked with disposable phishing portals.",
    "link": "Disproportionate frequency in unsolicited phishing emails.",
    "stream": "Commonly registered for fake streaming and credential theft.",
    "bid": "High association with deceptive prize and lottery lures.",
    "racing": "Abused in bulk automated spam and deceptive redirects.",
    "fit": "Frequently flagged in health lure spam and scam landing sites."
}

# Highly vetted, officially regulated, or restricted TLDs
TRUSTED_TLDS: Dict[str, str] = {
    "gov": "Strictly restricted government agency domain.",
    "edu": "Accredited educational institution domain.",
    "mil": "Strictly restricted military domain.",
    "bank": "Verified financial institution domain requiring strict charter proof.",
    "int": "Intergovernmental treaty organization domain."
}

# Registry of URL Shortening Services
SHORTENERS: Dict[str, str] = {
    "bit.ly": "Bitly Link Shortener",
    "goo.gl": "Google URL Shortener (Legacy)",
    "tinyurl.com": "TinyURL Redirection Service",
    "ow.ly": "Hootsuite Ow.ly Shortener",
    "t.co": "Twitter / X Link Wrapper",
    "is.gd": "is.gd Anonymous Shortener",
    "buff.ly": "Buffer Shortening Service",
    "adf.ly": "AdFly Monetized Redirector",
    "bit.do": "Bit.do Link Redirection",
    "cutt.ly": "Cuttly URL Shortener",
    "rb.gy": "Rebrandly Short Links",
    "shorturl.at": "ShortURL Service",
    "tiny.cc": "Tiny.cc URL Shortener",
    "lnkd.in": "LinkedIn Link Wrapper",
    "rebrandly.com": "Rebrandly Custom Shortener"
}


@dataclass
class TLDAnalysis:
    """Findings from TLD reputation lookup."""
    tld: str
    risk_level: str  # TRUSTED, NEUTRAL, SUSPICIOUS, HIGH, CRITICAL
    abuse_reason: Optional[str]
    is_shortener: bool
    shortener_name: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tld": self.tld,
            "risk_level": self.risk_level,
            "abuse_reason": self.abuse_reason,
            "is_shortener": self.is_shortener,
            "shortener_name": self.shortener_name,
        }


def extract_tld(url_or_domain: str) -> str:
    """Extract the primary Top-Level Domain extension."""
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return ""

    parts = hostname.split(".")
    if len(parts) >= 2:
        return parts[-1]
    return ""


def evaluate_tld_reputation(url_or_domain: str) -> TLDAnalysis:
    """
    Perform intelligence lookup on the URL's Top-Level Domain and shortener status.
    """
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower()

    tld = extract_tld(hostname)
    shortener_match = SHORTENERS.get(hostname)
    is_short = shortener_match is not None

    if not tld:
        return TLDAnalysis(
            tld="",
            risk_level="NEUTRAL",
            abuse_reason=None,
            is_shortener=is_short,
            shortener_name=shortener_match
        )

    if tld in TRUSTED_TLDS:
        return TLDAnalysis(
            tld=tld,
            risk_level="TRUSTED",
            abuse_reason=TRUSTED_TLDS[tld],
            is_shortener=is_short,
            shortener_name=shortener_match
        )

    if tld in ("zip", "mov", "icu", "top"):
        return TLDAnalysis(
            tld=tld,
            risk_level="CRITICAL",
            abuse_reason=HIGH_RISK_TLDS[tld],
            is_shortener=is_short,
            shortener_name=shortener_match
        )

    if tld in HIGH_RISK_TLDS:
        return TLDAnalysis(
            tld=tld,
            risk_level="HIGH",
            abuse_reason=HIGH_RISK_TLDS[tld],
            is_shortener=is_short,
            shortener_name=shortener_match
        )

    if is_short:
        return TLDAnalysis(
            tld=tld,
            risk_level="SUSPICIOUS",
            abuse_reason="URL Shortener conceals final landing destination.",
            is_shortener=True,
            shortener_name=shortener_match
        )

    return TLDAnalysis(
        tld=tld,
        risk_level="NEUTRAL",
        abuse_reason="Standard commercial or geographic Top-Level Domain.",
        is_shortener=False,
        shortener_name=None
    )
