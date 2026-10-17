"""
PhishEye Command-Line Interface (CLI).

Provides terminal-based URL scanning, forensic vector inspection, and batch threat
auditing with rich color formatting.
"""

import argparse
import json
import sys
from typing import List, Optional

from .detector import PhishEyeDetector
from .entropy import analyze_domain_entropy
from .homograph import analyze_homograph
from .security import defang_url, is_defanged, refang_url
from .tld_intel import evaluate_tld_reputation
from .typosquat import check_typosquatting

# ANSI color codes for terminal rendering
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"


def print_banner():
    banner = f"""{COLOR_CYAN}{COLOR_BOLD}
    ╔═══════════════════════════════════════════════╗
    ║        🛡️  PHISHEYE SECURITY CLI v2.1.0        ║
    ║   Real-Time ML-Powered Threat Intelligence    ║
    ╚═══════════════════════════════════════════════╝{COLOR_RESET}"""
    print(banner)


def scan_single_url(
    url: str,
    detector: PhishEyeDetector,
    as_json: bool = False,
    defang_output: bool = False,
    run_all: bool = True
) -> int:
    """Analyze a single URL and print findings."""
    raw_target = url.strip()
    if is_defanged(raw_target):
        active_url = refang_url(raw_target)
    else:
        active_url = raw_target

    scan = detector.predict(active_url)
    display_url = defang_url(active_url) if defang_output else active_url

    output_data = {
        "url": display_url,
        "raw_url": active_url,
        "verdict": scan.verdict,
        "is_phishing": scan.is_phishing,
        "confidence": scan.confidence,
        "risk_score": scan.risk_score,
        "risk_indicators": scan.risk_indicators,
        "safe_indicators": scan.safe_indicators,
    }

    if run_all:
        output_data["homograph"] = analyze_homograph(active_url).to_dict()
        output_data["typosquat"] = check_typosquatting(active_url).to_dict()
        output_data["entropy"] = analyze_domain_entropy(active_url).to_dict()
        output_data["tld"] = evaluate_tld_reputation(active_url).to_dict()

    if as_json:
        print(json.dumps(output_data, indent=2))
        return 1 if scan.is_phishing else 0

    # Formatted terminal display
    print(f"\n{COLOR_BOLD}Target URL:{COLOR_RESET} {display_url}")
    if scan.is_phishing:
        print(f"{COLOR_BOLD}Verdict:{COLOR_RESET}    {COLOR_RED}{COLOR_BOLD}🚨 {scan.verdict.upper()} (Risk: {scan.confidence:.2f}%){COLOR_RESET}")
    else:
        print(f"{COLOR_BOLD}Verdict:{COLOR_RESET}    {COLOR_GREEN}{COLOR_BOLD}✅ {scan.verdict.upper()} (Confidence: {scan.confidence:.2f}%){COLOR_RESET}")

    if scan.risk_indicators:
        print(f"\n{COLOR_YELLOW}{COLOR_BOLD}Risk Indicators Triggered:{COLOR_RESET}")
        for r in scan.risk_indicators:
            print(f"  {COLOR_RED}•{COLOR_RESET} {r}")

    if run_all:
        typo = output_data["typosquat"]
        if typo["is_impersonating"]:
            print(f"\n{COLOR_RED}{COLOR_BOLD}⚠️ Brand Impersonation Alert:{COLOR_RESET} {typo['details']}")

        homo = output_data["homograph"]
        if homo["is_homograph"]:
            print(f"{COLOR_RED}{COLOR_BOLD}⚠️ Homograph Spoofing Alert:{COLOR_RESET} {homo['risk_summary']}")

        ent = output_data["entropy"]
        if ent["is_dga_candidate"]:
            print(f"{COLOR_RED}{COLOR_BOLD}⚠️ DGA Randomness Alert:{COLOR_RESET} Entropy {ent['entropy']} ({ent['description']})")

        tld_info = output_data["tld"]
        if tld_info["risk_level"] in ("HIGH", "CRITICAL"):
            print(f"{COLOR_RED}{COLOR_BOLD}⚠️ TLD Risk Warning:{COLOR_RESET} .{tld_info['tld']} ({tld_info['abuse_reason']})")

    print(f"\n{COLOR_DIM}───────────────────────────────────────────────{COLOR_RESET}")
    return 1 if scan.is_phishing else 0


def scan_batch_file(
    file_path: str,
    detector: PhishEyeDetector,
    output_path: Optional[str] = None,
    as_json: bool = False,
    defang_output: bool = False,
    run_all: bool = True
) -> int:
    """Scan multiple URLs from a text file or CSV."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    results = []
    phishing_count = 0

    print(f"\n{COLOR_CYAN}Loaded {len(lines)} URLs from {file_path}. Commencing scan...{COLOR_RESET}\n")

    for idx, raw_url in enumerate(lines, 1):
        active_url = refang_url(raw_url) if is_defanged(raw_url) else raw_url
        scan = detector.predict(active_url)
        display_url = defang_url(active_url) if defang_output else active_url

        if scan.is_phishing:
            phishing_count += 1
            status_tag = f"{COLOR_RED}[PHISHING]{COLOR_RESET}"
        else:
            status_tag = f"{COLOR_GREEN}[SAFE]{COLOR_RESET}"

        item = {
            "index": idx,
            "url": display_url,
            "verdict": scan.verdict,
            "is_phishing": scan.is_phishing,
            "confidence": scan.confidence,
            "risk_score": scan.risk_score,
            "risk_indicators": scan.risk_indicators,
        }

        if run_all:
            item["homograph"] = analyze_homograph(active_url).is_homograph
            item["typosquat"] = check_typosquatting(active_url).is_impersonating
            item["entropy"] = round(analyze_domain_entropy(active_url).entropy, 3)
            item["tld_risk"] = evaluate_tld_reputation(active_url).risk_level

        results.append(item)
        if not as_json:
            print(f"[{idx:03d}/{len(lines):03d}] {status_tag} {scan.confidence:5.1f}% | {display_url[:60]}")

    safe_count = len(lines) - phishing_count

    if not as_json:
        print(f"\n{COLOR_BOLD}Scan Summary:{COLOR_RESET}")
        print(f"  Total Scanned:  {len(lines)}")
        print(f"  {COLOR_RED}Phishing Flags:{COLOR_RESET} {phishing_count}")
        print(f"  {COLOR_GREEN}Safe Passed:   {COLOR_RESET} {safe_count}")

    if output_path:
        if output_path.endswith(".csv"):
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"\n{COLOR_CYAN}Exported CSV report to: {output_path}{COLOR_RESET}")
        else:
            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(results, out, indent=2)
            print(f"\n{COLOR_CYAN}Exported JSON report to: {output_path}{COLOR_RESET}")

    if as_json and not output_path:
        print(json.dumps(results, indent=2))

    return 1 if phishing_count > 0 else 0


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PhishEye - Real-time ML-powered Phishing Threat Detection CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", type=str, help="Single URL to inspect")
    group.add_argument("-f", "--file", type=str, help="Path to text or CSV file containing URLs")

    parser.add_argument("-o", "--output", type=str, help="Path to export report (JSON or CSV)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--defang", action="store_true", help="Defang URLs in terminal output")
    parser.add_argument("--no-banner", action="store_true", help="Suppress introductory banner")
    parser.add_argument("--all", action="store_true", default=True, help="Execute deep forensic audits (homograph, typosquatting, entropy, TLD)")

    parsed_args = parser.parse_args(args)

    if not parsed_args.json and not parsed_args.no_banner:
        print_banner()

    detector = PhishEyeDetector()

    if parsed_args.url:
        return scan_single_url(
            url=parsed_args.url,
            detector=detector,
            as_json=parsed_args.json,
            defang_output=parsed_args.defang,
            run_all=parsed_args.all
        )
    elif parsed_args.file:
        return scan_batch_file(
            file_path=parsed_args.file,
            detector=detector,
            output_path=parsed_args.output,
            as_json=parsed_args.json,
            defang_output=parsed_args.defang,
            run_all=parsed_args.all
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

