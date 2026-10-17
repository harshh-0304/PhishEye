"""
PhishEye Model Inference and Threat Classification Engine.

Wraps the pre-trained Scikit-Learn Random Forest model to provide
high-level prediction APIs, confidence scoring, probability calculations,
and structured scan result objects.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd

from .extractor import URLFeatureExtractor, FEATURE_NAMES


@dataclass
class ScanResult:
    """Structured result of a URL phishing inspection."""
    url: str
    is_phishing: bool
    verdict: str
    confidence: float
    risk_score: float
    probabilities: Dict[str, float]
    features: Dict[str, int]
    risk_indicators: List[str] = field(default_factory=list)
    safe_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to a serializable dictionary."""
        return {
            "url": self.url,
            "verdict": self.verdict,
            "is_phishing": self.is_phishing,
            "confidence": round(self.confidence, 2),
            "risk_score": round(self.risk_score, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "risk_indicators": self.risk_indicators,
            "safe_indicators": self.safe_indicators,
            "features": self.features,
        }


class PhishEyeDetector:
    """
    High-level threat detector wrapping the ML classifier and feature extractor.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names_path: Optional[str] = None
    ):
        self.model_path = model_path or self._find_artifact("phishing_detector_model.pkl")
        self.feature_names_path = feature_names_path or self._find_artifact("feature_names.pkl")

        self.model = joblib.load(self.model_path)
        self.feature_names = joblib.load(self.feature_names_path)
        self.extractor = URLFeatureExtractor(feature_order=self.feature_names)

    @staticmethod
    def _find_artifact(filename: str) -> str:
        """Find artifact relative to current module or working directory."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", filename),
            os.path.join(here, filename),
            filename,
            os.path.join("PhishEye", filename),
            os.path.join(os.getcwd(), filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)
        raise FileNotFoundError(f"Required artifact '{filename}' could not be located in candidates: {candidates}")

    def _generate_indicators(self, features: Dict[str, int], url: str) -> Tuple[List[str], List[str]]:
        """Extract human-readable security indicators from extracted features."""
        risks: List[str] = []
        safes: List[str] = []

        if features.get("SSLfinal_State") == -1:
            risks.append("No HTTPS Protocol - Connection lacks SSL/TLS encryption.")
        else:
            safes.append("Valid HTTPS Encryption - Secure communication protocol verified.")

        if features.get("having_At_Symbol") == -1:
            risks.append("'@' Character Injection - Used to obscure true destination authority.")
        else:
            safes.append("Clean Authority String - No deceptive symbol injection.")

        if features.get("Shortining_Service") == -1:
            risks.append("URL Shortener Detected - Masks final landing domain.")
        else:
            safes.append("Direct Destination - Domain does not route through known shorteners.")

        if features.get("having_IPhaving_IP_Address") == -1:
            risks.append("IP Address Destination - Hostname resolves to raw IP instead of registered domain.")
        else:
            safes.append("Named Domain - Uses registered DNS host.")

        if features.get("Prefix_Suffix") == -1:
            risks.append("Hyphenated Domain - Hyphen in hostname indicates brand spoofing attempt.")

        if features.get("having_Sub_Domain") == -1:
            risks.append("Excessive Subdomains - Multi-level domain structure (>2 subdomains).")

        if features.get("URLURL_Length") == -1:
            risks.append(f"Suspicious URL Length ({len(url)} chars) - Abnormally long path string.")
        elif features.get("URLURL_Length") == 1:
            safes.append(f"Standard URL Length ({len(url)} chars).")

        if features.get("HTTPS_token") == -1:
            risks.append("Deceptive 'HTTPS' Token - Appears in hostname or path but connection is HTTP.")

        return risks, safes

    def predict(self, url: str) -> ScanResult:
        """
        Analyze a single URL and return detailed classification and metrics.
        """
        features_dict = self.extractor.extract(url)
        df_row = pd.DataFrame([[features_dict[f] for f in self.feature_names]], columns=self.feature_names)

        # Model prediction: 0 is Phishing, 1 is Safe
        pred = int(self.model.predict(df_row)[0])
        proba = self.model.predict_proba(df_row)[0]

        phishing_prob = float(proba[0])
        safe_prob = float(proba[1]) if len(proba) > 1 else (1.0 - phishing_prob)

        is_phishing = (pred == 0)
        verdict = "Phishing Detected" if is_phishing else "Safe Domain"
        confidence = (phishing_prob if is_phishing else safe_prob) * 100.0
        risk_score = phishing_prob

        risks, safes = self._generate_indicators(features_dict, url)

        return ScanResult(
            url=url,
            is_phishing=is_phishing,
            verdict=verdict,
            confidence=confidence,
            risk_score=risk_score,
            probabilities={"phishing": phishing_prob, "safe": safe_prob},
            features=features_dict,
            risk_indicators=risks,
            safe_indicators=safes
        )

    def predict_batch(self, urls: List[str]) -> List[ScanResult]:
        """Analyze a list of URLs sequentially."""
        return [self.predict(u) for u in urls if u and u.strip()]
