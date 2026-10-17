"""
PhishEye IDN Homograph & Punycode Threat Detection Engine.

Detects Internationalized Domain Name (IDN) homograph attacks where malicious
actors substitute visually identical glyphs (Cyrillic, Greek, Cherokee, etc.)
for standard ASCII Latin characters to deceive victims.
"""

from dataclasses import dataclass, field
import unicodedata
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Character map of common Cyrillic / Greek glyphs used to spoof Latin characters
HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic small letters
    "\u0430": "a",  # Cyrillic small a
    "\u0441": "c",  # Cyrillic small es (c)
    "\u0435": "e",  # Cyrillic small ie (e)
    "\u0456": "i",  # Cyrillic small Byelorussian-Ukrainian i
    "\u0458": "j",  # Cyrillic small je (j)
    "\u043e": "o",  # Cyrillic small o
    "\u0440": "p",  # Cyrillic small er (p)
    "\u0455": "s",  # Cyrillic small dze (s)
    "\u0443": "y",  # Cyrillic small u (y)
    "\u0445": "x",  # Cyrillic small ha (x)
    # Greek small letters
    "\u03b1": "a",  # Greek small alpha
    "\u03bf": "o",  # Greek small omicron
    "\u03bd": "v",  # Greek small nu
}


@dataclass
class HomographReport:
    """Report detailing IDN homograph and punycode spoofing indicators."""
    domain: str
    is_homograph: bool
    is_punycode: bool
    punycode_decoded: Optional[str]
    has_mixed_scripts: bool
    confusable_glyphs: List[Dict[str, str]] = field(default_factory=list)
    risk_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "is_homograph": self.is_homograph,
            "is_punycode": self.is_punycode,
            "punycode_decoded": self.punycode_decoded,
            "has_mixed_scripts": self.has_mixed_scripts,
            "confusable_glyphs": self.confusable_glyphs,
            "risk_summary": self.risk_summary,
        }


def extract_domain(url_or_domain: str) -> str:
    """Extract clean domain/hostname from a full URL or domain string."""
    cleaned = (url_or_domain or "").strip()
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    host = parsed.hostname or ""
    return host.lower()


def detect_punycode(domain: str) -> bool:
    """Check if domain contains Punycode representation prefix."""
    return "xn--" in domain.lower()


def decode_punycode(domain: str) -> Optional[str]:
    """Safely decode punycode domain to Unicode representation."""
    if not detect_punycode(domain):
        return None
    try:
        return domain.encode("ascii").decode("idna")
    except Exception:
        return None


def detect_mixed_scripts(text: str) -> bool:
    """Determine if a text contains glyphs from multiple conflicting writing systems."""
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char, "")
            script = name.split()[0] if name else "UNKNOWN"
            scripts.add(script)
        except Exception:
            pass
    # If more than one alphabetical script (e.g. LATIN and CYRILLIC), mixed script attack detected
    return len(scripts) > 1


def find_confusables(text: str) -> List[Dict[str, str]]:
    """Scan text for known homoglyph characters mimicking Latin letters."""
    found = []
    for char in text:
        if char in HOMOGLYPH_MAP:
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = "UNKNOWN"
            found.append({
                "character": char,
                "codepoint": f"U+{ord(char):04X}",
                "latin_equivalent": HOMOGLYPH_MAP[char],
                "unicode_name": char_name
            })
    return found


def analyze_homograph(url_or_domain: str) -> HomographReport:
    """
    Perform deep homograph and punycode analysis on a given URL or domain.
    """
    domain = extract_domain(url_or_domain)
    is_puny = detect_punycode(domain)
    decoded = decode_punycode(domain) if is_puny else None

    target_text = decoded if decoded else domain
    confusables = find_confusables(target_text)
    mixed = detect_mixed_scripts(target_text)

    is_homograph = is_puny or mixed or len(confusables) > 0

    if is_puny and (mixed or confusables):
        summary = "CRITICAL: Punycode domain spoofing detected with disguised foreign homoglyphs."
    elif is_puny:
        summary = "WARNING: Punycode (IDN) domain detected. Verify decoded target authority."
    elif mixed:
        summary = "CRITICAL: Mixed-script Unicode detected; letters from multiple scripts intermingled."
    elif confusables:
        summary = "SUSPICIOUS: Confusable character glyphs identified mimicking Latin alphabet."
    else:
        summary = "CLEAN: Standard ASCII character set without homograph indicators."

    return HomographReport(
        domain=domain,
        is_homograph=is_homograph,
        is_punycode=is_puny,
        punycode_decoded=decoded,
        has_mixed_scripts=mixed,
        confusable_glyphs=confusables,
        risk_summary=summary
    )
