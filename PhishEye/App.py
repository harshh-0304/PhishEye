import os
import sys
from urllib.parse import urlparse
import pandas as pd
import streamlit as st

# Ensure phisheye package is importable when run from root or PhishEye directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phisheye.detector import PhishEyeDetector, ScanResult
from phisheye.extractor import FEATURE_NAMES

# --- 1. Initialize PhishEye Detection Engine ---
@st.cache_resource
def load_detector():
    return PhishEyeDetector()

detector = load_detector()
feature_names = detector.feature_names
model = detector.model

def get_features_from_url(url):
    """Wrapper using URLFeatureExtractor for backward compatibility."""
    return detector.extractor.to_dataframe(url)

# --- 2. Streamlit UI Configuration & Styling ---
st.set_page_config(
    page_title="PhishEye Cyber Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    candidates = [
        file_name,
        os.path.join(os.path.dirname(__file__), file_name),
        os.path.join(os.path.dirname(__file__), "style.css")
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            return

local_css("style.css")


st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <span style='background: rgba(14, 165, 233, 0.15); color: #0ea5e9; border: 1px solid rgba(14, 165, 233, 0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>AI Security Engine v2.0</span>
        <h1 style='color: #f8fafc; letter-spacing: -0.02em; margin-top: 0.75rem; margin-bottom: 0.5rem;'>🛡️ PhishEye URL Inspector</h1>
        <p style='color: #94a3b8; font-size: 1.05rem; max-width: 600px; margin: 0 auto;'>Instant Machine Learning threat detection for suspicious domains & URLs.</p>
    </div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/phishing.png", width=70)
    st.title("PhishEye Cyber Guard")
    st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;'>
            <p style='color: #e2e8f0; font-size: 0.85rem; margin: 0;'>
                Real-time threat detection trained on <strong>30 structural URL features</strong> using Random Forest classification.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### Stack & Model")
    st.caption("• Framework: Streamlit + Python\n• Model: Scikit-learn Classifier\n• Features: 30 URL vectors")
    st.markdown("---")
    st.markdown("[🔗 GitHub Repository](https://github.com/harshh-0304/PhishEye)")

tab_single, tab_batch = st.tabs(["🔍 Single URL Inspector", "📂 Batch URL Scanner"])

with tab_single:
    with st.container():
        st.markdown("**Quick Preset Test Cases:**")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    default_url = "https://security-update-bankofamerica.com/login?id=123"
    if "current_url" not in st.session_state:
        st.session_state["current_url"] = default_url

    if btn_col1.button("🚨 Phishing Sample", use_container_width=True):
        st.session_state["current_url"] = "http://paypal-security-account-update.login-auth.com/verify@user"
    if btn_col2.button("✅ Safe Domain Sample", use_container_width=True):
        st.session_state["current_url"] = "https://google.com"
    if btn_col3.button("⚠️ Shortened Link", use_container_width=True):
        st.session_state["current_url"] = "http://bit.ly/secure-account-login"

    user_input = st.text_input("Enter URL to analyze:", value=st.session_state["current_url"], help="Try entering a suspicious URL or a legitimate one like https://google.com")

st.markdown("---")

if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Analyze URL", type="primary"):
    if user_input:
        st.info("Analyzing the provided URL...")
        
        # Get features from the input URL
        input_features_df = get_features_from_url(user_input)

        # Make prediction
        prediction = model.predict(input_features_df)[0]
        prediction_proba = model.predict_proba(input_features_df)[0]
        confidence = prediction_proba[prediction] * 100
        status_label = "Phishing Detected" if prediction == 0 else "Safe Domain"

        st.session_state["history"].insert(0, {
            "URL": user_input,
            "Verdict": status_label,
            "Confidence": f"{confidence:.2f}%"
        })

        st.markdown("### Analysis Result:")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 0: # 0 for phishing
                st.markdown('<div class="status-card error"><h3>🚨 PHISHING DETECTED!</h3><p>Proceed with extreme caution.</p></div>', unsafe_allow_html=True)
            else: # 1 for safe
                st.markdown('<div class="status-card success"><h3>✅ SAFE</h3><p>No Phishing Detected.</p></div>', unsafe_allow_html=True)

        with col2:
            confidence = prediction_proba[prediction] * 100
            bar_color = "#ef4444" if prediction == 0 else "#22c55e"
            st.markdown(f'''
                <div class="status-card" style="text-align: center;">
                    <h4>Confidence Score</h4>
                    <h2 class="mono-text" style="color: {bar_color}; font-size: 2rem; margin-bottom: 0.5rem;">{confidence:.2f}%</h2>
                    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;">
                        <div style="background: {bar_color}; width: {confidence:.2f}%; height: 100%; border-radius: 9999px; transition: width 0.5s ease;"></div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        # Deep Forensic Telemetry
        from phisheye.entropy import analyze_domain_entropy
        from phisheye.homograph import analyze_homograph
        from phisheye.typosquat import check_typosquatting
        from phisheye.tld_intel import evaluate_tld_reputation

        entropy_report = analyze_domain_entropy(user_input)
        typo_report = check_typosquatting(user_input)
        homo_report = analyze_homograph(user_input)
        tld_report = evaluate_tld_reputation(user_input)

        st.markdown("---")
        st.markdown("### Domain Health & Forensic Telemetry:")
        m1, m2, m3, m4 = st.columns(4)
        ssl_val = "HTTPS Active" if input_features_df['SSLfinal_State'].iloc[0] == 1 else "No HTTPS"
        url_len_val = f"{len(user_input)} Chars"
        short_val = "Detected" if input_features_df['Shortining_Service'].iloc[0] == -1 else "Clean"
        m1.metric("SSL Protocol", ssl_val, delta="Secure" if ssl_val == "HTTPS Active" else "-Risk", delta_color="normal" if ssl_val == "HTTPS Active" else "inverse")
        m2.metric("URL Length", url_len_val, delta="Normal" if len(user_input) < 54 else "-Long", delta_color="normal" if len(user_input) < 54 else "inverse")
        m3.metric("URL Shortener", short_val, delta="Clean" if short_val == "Clean" else "-Flagged", delta_color="normal" if short_val == "Clean" else "inverse")
        m4.metric("Shannon Entropy", f"{entropy_report.entropy:.2f}", delta=entropy_report.entropy_category, delta_color="inverse" if entropy_report.is_dga_candidate else "normal")

        # High-Risk Threat Signal Callouts
        if typo_report.is_impersonating:
            st.markdown(f"""
                <div style='background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 12px 16px; margin-top: 12px; margin-bottom: 12px;'>
                    <h4 style='color: #ef4444; margin: 0 0 4px 0;'>🚨 Brand Impersonation Alert ({typo_report.attack_vector})</h4>
                    <p style='color: #fca5a5; margin: 0; font-size: 0.9rem;'>{typo_report.details} Target brand: <strong>{typo_report.impersonated_brand}</strong> (Official: {typo_report.legitimate_domain})</p>
                </div>
            """, unsafe_allow_html=True)

        if homo_report.is_homograph:
            st.markdown(f"""
                <div style='background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;'>
                    <h4 style='color: #f59e0b; margin: 0 0 4px 0;'>⚠️ IDN Homograph / Punycode Alert</h4>
                    <p style='color: #fde68a; margin: 0; font-size: 0.9rem;'>{homo_report.risk_summary}</p>
                </div>
            """, unsafe_allow_html=True)

        if tld_report.risk_level in ("HIGH", "CRITICAL"):
            st.markdown(f"""
                <div style='background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;'>
                    <h4 style='color: #f87171; margin: 0 0 4px 0;'>⚠️ Suspicious Top-Level Domain (.{tld_report.tld})</h4>
                    <p style='color: #fca5a5; margin: 0; font-size: 0.9rem;'>{tld_report.abuse_reason}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Reasoning & Key Indicators:")

        # Provide reasoning based on the input features and their importance
        if prediction == 0: # If classified as phishing
            st.error("🚨 Warning: This URL exhibits characteristics strongly associated with phishing sites.")
            
            # Highlight specific features that contributed to the phishing detection
            if input_features_df['SSLfinal_State'].iloc[0] == -1:
                st.markdown("<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ No HTTPS Protocol</span> &mdash; Connection lacks SSL encryption.</div>", unsafe_allow_html=True)
            if input_features_df['having_At_Symbol'].iloc[0] == -1:
                st.markdown("<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ '@' Character Injection</span> &mdash; Used to obscure true domain destination.</div>", unsafe_allow_html=True)
            if input_features_df['URLURL_Length'].iloc[0] == -1:
                st.markdown(f"<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ Suspicious URL Length ({len(user_input)} chars)</span> &mdash; Unusually long path string.</div>", unsafe_allow_html=True)
            if input_features_df['Shortining_Service'].iloc[0] == -1:
                st.markdown("<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ URL Shortener Service</span> &mdash; Conceals final landing page.</div>", unsafe_allow_html=True)
            if input_features_df['having_Sub_Domain'].iloc[0] == -1 or input_features_df['having_Sub_Domain'].iloc[0] == 0:
                st.markdown("<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ Multiple Subdomains</span> &mdash; Can impersonate trusted brands.</div>", unsafe_allow_html=True)
            if input_features_df['Prefix_Suffix'].iloc[0] == -1:
                st.markdown("<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #ef4444; font-weight: 600;'>❌ Hyphenated Domain</span> &mdash; Common typo-squatting pattern.</div>", unsafe_allow_html=True)
            
            st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: 10px;'><em>Note: ML features extracted directly from real-time URL string analysis.</em></p>", unsafe_allow_html=True)
            
        else: # If classified as safe
            st.success("✅ Legitimacy Verified: URL passes security inspection checks.")
            if input_features_df['SSLfinal_State'].iloc[0] == 1:
                st.markdown("<div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #22c55e; font-weight: 600;'>✅ Valid HTTPS Encryption</span> &mdash; Secure communication protocol verified.</div>", unsafe_allow_html=True)
            if input_features_df['having_At_Symbol'].iloc[0] == 1:
                st.markdown("<div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #22c55e; font-weight: 600;'>✅ Standard Authority String</span> &mdash; No deceptive symbol syntax.</div>", unsafe_allow_html=True)
            if input_features_df['URLURL_Length'].iloc[0] == 1 or input_features_df['URLURL_Length'].iloc[0] == 0:
                st.markdown(f"<div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;'><span style='color: #22c55e; font-weight: 600;'>✅ Optimal URL Length ({len(user_input)} chars)</span> &mdash; Normal domain structure.</div>", unsafe_allow_html=True)
            st.markdown("- *Note: Analysis is based on a set of extracted URL features.*")



        with st.expander("🔍 Detailed 30-Feature ML Extraction Vector"):
            df_display = input_features_df.T.reset_index()
            df_display.columns = ["Feature Name", "Extracted Value (-1: Risk, 0: Neutral, 1: Safe)"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        if len(st.session_state["history"]) > 1:
            with st.expander(f"📜 Session Scan History ({len(st.session_state['history'])} URLs scanned)"):
                st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)
        
    else:
        st.warning("Please enter a URL to analyze.")

with tab_batch:
    st.markdown("### 📂 High-Throughput Batch URL Inspection")
    st.caption("Upload a `.txt` or `.csv` file containing URLs, or paste multiple targets to perform bulk classification.")

    sample_batch_text = (
        "https://google.com\n"
        "http://paypal-security-account.login-auth.com/verify@user\n"
        "https://github.com\n"
        "http://bit.ly/secure-account-login\n"
        "https://microsoft.com\n"
        "http://192.168.1.1/admin-login"
    )

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload URL list (.txt or .csv)", type=["txt", "csv"])
    with col_b2:
        if st.button("Load Predefined Batch Sample", use_container_width=True):
            st.session_state["batch_input_text"] = sample_batch_text

    batch_text = st.text_area(
        "Paste URLs (one per line):",
        value=st.session_state.get("batch_input_text", ""),
        height=140,
        placeholder="https://example.com\nhttps://suspicious-site.net"
    )

    if st.button("🚀 Run Batch Security Scan", type="primary"):
        urls_to_scan = []
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            urls_to_scan = [u.strip() for u in content.splitlines() if u.strip() and not u.strip().startswith("#")]
        elif batch_text.strip():
            urls_to_scan = [u.strip() for u in batch_text.splitlines() if u.strip() and not u.strip().startswith("#")]

        if not urls_to_scan:
            st.warning("Please upload a file or paste at least one URL to scan.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            batch_results = []

            for i, u in enumerate(urls_to_scan):
                status_text.text(f"Scanning target {i + 1} of {len(urls_to_scan)}: {u[:40]}...")
                scan = detector.predict(u)
                batch_results.append({
                    "Target URL": u,
                    "Verdict": "🚨 Phishing" if scan.is_phishing else "✅ Safe",
                    "Confidence": f"{scan.confidence:.2f}%",
                    "Risk Score": f"{scan.risk_score:.3f}",
                    "Key Threat Signal": scan.risk_indicators[0] if scan.risk_indicators else "None (Clean)"
                })
                progress_bar.progress((i + 1) / len(urls_to_scan))

            status_text.empty()
            progress_bar.empty()

            df_batch = pd.DataFrame(batch_results)
            phish_total = sum(1 for r in batch_results if "Phishing" in r["Verdict"])
            safe_total = len(batch_results) - phish_total

            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("Total Scanned", len(batch_results))
            bm2.metric("Phishing Flagged", phish_total, delta=f"-{phish_total}" if phish_total > 0 else "0", delta_color="inverse")
            bm3.metric("Legitimate Verified", safe_total, delta=f"+{safe_total}")

            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            csv_data = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Batch Results (CSV)",
                data=csv_data,
                file_name="phisheye_batch_scan_results.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")
st.markdown("""
    <div style='text-align: center; margin-top: 40px; margin-bottom: 20px; opacity: 0.75; font-size: 0.85rem;'>
        <p style='margin-bottom: 4px;'><strong>PhishEye Real-Time Phishing Detector v2.0</strong> &bull; Developed by <a href='https://github.com/harshh-0304' target='_blank' style='color: #0ea5e9; text-decoration: none;'>harshh-0304</a></p>
        <p style='color: #64748b; margin: 0;'>Powered by Streamlit & ML Classification | MIT License</p>
    </div>
""", unsafe_allow_html=True)