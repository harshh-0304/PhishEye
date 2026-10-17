"""
PhishEye Multi-Format Forensic Threat Report Generator.

Transforms URL scan results and forensic telemetry into publication-ready
incident reports formatted as Markdown, JSON, CSV, or standalone HTML snippets.
"""

from datetime import datetime, timezone
import json
from typing import List, Optional
import io
import pandas as pd

from .detector import ScanResult
from .security import defang_url


def generate_markdown_report(scan: ScanResult, analyst_notes: Optional[str] = None) -> str:
    """Generate a GitHub-flavored Markdown incident report for a scanned URL."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    defanged = defang_url(scan.url)
    verdict_emoji = "🚨" if scan.is_phishing else "✅"
    verdict_badge = f"**{verdict_emoji} {scan.verdict.upper()}**"

    lines = [
        f"# 🛡️ PhishEye Threat Intelligence Report",
        f"",
        f"- **Generated:** `{now_str}`",
        f"- **Target URL:** `{defanged}` *(Defanged for safety)*",
        f"- **Security Verdict:** {verdict_badge}",
        f"- **Model Confidence:** `{scan.confidence:.2f}%`",
        f"- **Calculated Risk Score:** `{scan.risk_score:.4f}`",
        f"",
        f"---",
        f"",
        f"## 🔍 Primary Indicators",
        f""
    ]

    if scan.risk_indicators:
        lines.append("### ⚠️ Flagged Threat Signals:")
        for r in scan.risk_indicators:
            lines.append(f"- ❌ {r}")
        lines.append("")

    if scan.safe_indicators:
        lines.append("### ✅ Verified Integrity Signals:")
        for s in scan.safe_indicators:
            lines.append(f"- ✔️ {s}")
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 📊 30-Dimension Feature Vector",
        f"",
        f"| Feature Name | Extracted Value | Interpretation |",
        f"| :--- | :---: | :--- |"
    ])

    for feat_name, val in scan.features.items():
        interp = "Phishing Risk (-1)" if val == -1 else ("Suspicious (0)" if val == 0 else "Legitimate (+1)")
        lines.append(f"| `{feat_name}` | `{val}` | {interp} |")

    if analyst_notes:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 📝 SOC Analyst Notes",
            f"",
            f"{analyst_notes}"
        ])

    lines.extend([
        f"",
        f"---",
        f"*Report generated automatically by PhishEye v2.1.0 AI Engine.*"
    ])

    return "\n".join(lines)


def generate_json_report(scan: ScanResult) -> str:
    """Generate structured JSON report representation."""
    report_dict = scan.to_dict()
    report_dict["generated_at"] = datetime.now(timezone.utc).isoformat()
    report_dict["defanged_url"] = defang_url(scan.url)
    return json.dumps(report_dict, indent=2)


def generate_csv_report(scans: List[ScanResult]) -> str:
    """Generate CSV string containing summary rows for a batch of scans."""
    rows = []
    for s in scans:
        rows.append({
            "url": defang_url(s.url),
            "verdict": s.verdict,
            "is_phishing": s.is_phishing,
            "confidence_percent": round(s.confidence, 2),
            "risk_score": round(s.risk_score, 4),
            "threat_signals_count": len(s.risk_indicators)
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
