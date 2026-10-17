"""
PhishEye Security and Threat Neutralization Utilities.

Provides URL defanging and refanging algorithms to ensure safe handling of
malicious URLs in incident response reports, SOC log files, and tickets without
risking accidental click-throughs.
"""

import re
from typing import Union


def defang_url(url: str, bracket_dots: bool = True) -> str:
    """
    Defang a URL or IP address so it is not accidentally clickable or resolved.

    Examples:
        https://evil.example.com/path -> hxxps[://]evil[.]example[.]com/path
        http://1.2.3.4 -> hxxp[://]1[.]2[.]3[.]4
    """
    if not url:
        return ""

    defanged = url.strip()

    # Defang scheme
    defanged = re.sub(r"^https://", "hxxps[://]", defanged, flags=re.IGNORECASE)
    defanged = re.sub(r"^http://", "hxxp[://]", defanged, flags=re.IGNORECASE)
    defanged = re.sub(r"^ftp://", "fxp[://]", defanged, flags=re.IGNORECASE)

    # Defang dots if requested
    if bracket_dots:
        defanged = defanged.replace(".", "[.]")

    return defanged


def refang_url(url: str) -> str:
    """
    Restore a defanged URL or IP address back to its standard executable syntax.

    Examples:
        hxxps[://]evil[.]example[.]com/path -> https://evil.example.com/path
        hxxp://evil.com -> http://evil.com
        192[.]168[.]1[.]1 -> 192.168.1.1
    """
    if not url:
        return ""

    refanged = url.strip()

    # Replace bracketed dots
    refanged = refanged.replace("[.]", ".")
    refanged = refanged.replace("(.)", ".")

    # Replace bracketed colons/slashes
    refanged = refanged.replace("[:]", ":")
    refanged = refanged.replace("[/]", "/")
    refanged = refanged.replace("[://]", "://")
    refanged = refanged.replace("(://)", "://")

    # Restore schemes
    refanged = re.sub(r"^hxxps://", "https://", refanged, flags=re.IGNORECASE)
    refanged = re.sub(r"^hxxp://", "http://", refanged, flags=re.IGNORECASE)
    refanged = re.sub(r"^fxp://", "ftp://", refanged, flags=re.IGNORECASE)

    return refanged


def is_defanged(url: str) -> bool:
    """Check if a URL contains common defanging markers."""
    if not url:
        return False
    markers = ["[.]", "(.)", "hxxp", "fxp", "[://]", "[:]"]
    return any(marker in url.lower() for marker in markers)
