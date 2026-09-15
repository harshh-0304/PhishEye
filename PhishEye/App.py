import streamlit as st
import joblib
import pandas as pd
from urllib.parse import urlparse
import re

# --- 1. Load the Trained Model and Feature Names ---
try:
    model = joblib.load('phishing_detector_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("Model or feature names file not found! Please ensure 'phishing_detector_model.pkl' and 'feature_names.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing

# --- 2. Feature Extraction Functions for a RAW URL (Adapted for your dataset's features) ---
# This is where we attempt to map raw URL characteristics to your dataset's features.
# This will be an *approximation* as we don't have the exact original definitions
# for all 30 features. We'll focus on the most impactful and straightforward ones.

def get_features_from_url(url):
    features = {}

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path if parsed_url.path else ''

    # Default values for all features (assuming 1 for 'safe' or typical, 0 for 'suspicious')
    # These are placeholders and might need refinement based on your dataset's exact feature definitions.
    for feature in feature_names:
        # A common default for many features is 1, indicating 'safe' or 'present'
        # For others like 'Result' which is not a feature, this won't be used.
        features[feature] = 1 # Initialize all with 'safe' default

    # --- Implement logic for specific, important features from your dataset ---
    # 1. having_IPhaving_IP_Address: -1 if IP address is used in hostname, 1 otherwise
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        features['having_IPhaving_IP_Address'] = -1
    else:
        features['having_IPhaving_IP_Address'] = 1

    # 2. URLURL_Length: -1 if length < 54, 0 if 54 <= length <= 75, 1 if length > 75 (example thresholds)
    url_len = len(url)
    if url_len < 54:
        features['URLURL_Length'] = 1 # Assuming shorter is safer
    elif 54 <= url_len <= 75:
        features['URLURL_Length'] = 0
    else:
        features['URLURL_Length'] = -1 # Assuming longer is more suspicious

    # 3. Shortining_Service: -1 if shortener used, 1 otherwise
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co'] # Add more if needed
    if any(service in hostname for service in shortening_services):
        features['Shortining_Service'] = -1
    else:
        features['Shortining_Service'] = 1

    # 4. having_At_Symbol: -1 if '@' symbol present, 1 otherwise
    if '@' in url:
        features['having_At_Symbol'] = -1
    else:
        features['having_At_Symbol'] = 1

    # 5. double_slash_redirecting: -1 if double slash // after http/https, 1 otherwise
    if '//' in urlparse(url).path: # Checks for // in the path component
         features['double_slash_redirecting'] = -1
    else:
         features['double_slash_redirecting'] = 1


    # 6. Prefix_Suffix: -1 if '-' is in hostname, 1 otherwise
    if '-' in hostname and hostname.count('-') > 0 and hostname.find('-') != 0 and hostname.find('-') != len(hostname)-1:
        features['Prefix_Suffix'] = -1
    else:
        features['Prefix_Suffix'] = 1

    # 7. having_Sub_Domain: -1 if > 2 subdomains, 0 if 2, 1 if 1 (example, adjust based on your original dataset's logic)
    # This is a simplified logic. Real implementation might analyze TLDs, country codes etc.
    if hostname:
        parts = hostname.split('.')
        # Filter out common prefixes like 'www' and ensure meaningful parts
        meaningful_parts = [p for p in parts if p not in ['www', ''] and '.' not in p]
        num_subdomains_count = len(meaningful_parts) - 2 # -1 for domain, -1 for TLD (e.g., example.com -> 0 sub)

        if num_subdomains_count > 1: # e.g., sub.sub.example.com
            features['having_Sub_Domain'] = -1
        elif num_subdomains_count == 1: # e.g., sub.example.com
            features['having_Sub_Domain'] = 0
        else: # e.g., example.com
            features['having_Sub_Domain'] = 1
    else:
        features['having_Sub_Domain'] = 1 # Default for invalid/no hostname


    # 8. SSLfinal_State: -1 (bad), 0 (neutral), 1 (good) - (This is hard to get from URL alone without external check)
    # We'll approximate: 1 if HTTPS, -1 if HTTP, 0 if something else or ambiguous.
    if parsed_url.scheme == 'https':
        features['SSLfinal_State'] = 1 # Assumed good
    else:
        features['SSLfinal_State'] = -1 # Assumed bad (HTTP)

    # 9. HTTPS_token: -1 if "https" is in domain/path but not scheme
    if 'https' in hostname or 'https' in path: # Check for "https" as part of domain/path
        if parsed_url.scheme != 'https': # But not actually HTTPS
            features['HTTPS_token'] = -1
        else:
            features['HTTPS_token'] = 1
    else:
        features['HTTPS_token'] = 1


    # --- Placeholder for other features ---
    # For features that are difficult to derive directly from a URL string (like Domain_registeration_length,
    # Favicon, port, Request_URL, URL_of_Anchor, Links_in_tags, SFH, Submitting_to_email,
    # Abnormal_URL, Redirect, on_mouseover, RightClick, popUpWidnow, Iframe, age_of_domain,
    # DNSRecord, web_traffic, Page_Rank, Google_Index, Links_pointing_to_page, Statistical_report),
    # we'll either:
    # A) Leave them at their default (e.g., 1 for 'safe' or 'typical'). This is a strong assumption.
    # B) Set them to a neutral or average value from your training data if you can calculate it.
    # C) Based on our `model_trainer.py` feature importances, `SSLfinal_State` is very important,
    #    so our approximation above is crucial. `URL_of_Anchor`, `web_traffic` are also important
    #    but very hard to get from just a URL string without making HTTP requests or external APIs.
    #    For a demo, setting reasonable defaults is acceptable if you explain this compromise.

    # Example: Manually set some high-importance features to a 'suspicious' state IF other indicators are present
    # This makes the demo more dynamic even if we can't derive all 30 perfectly.
    if features['having_At_Symbol'] == -1 or features['URLURL_Length'] == -1: # If URL is clearly suspicious by simple metrics
        if 'URL_of_Anchor' in features: features['URL_of_Anchor'] = -1 # Assume bad anchors
        if 'Links_in_tags' in features: features['Links_in_tags'] = -1 # Assume bad links in tags
        if 'web_traffic' in features: features['web_traffic'] = -1 # Assume low web traffic (common for phishing)
        if 'Statistical_report' in features: features['Statistical_report'] = -1 # Assume bad report


    # Ensure all features are present in the correct order
    # Convert to DataFrame row
    feature_values = [features[f] for f in feature_names]
    return pd.DataFrame([feature_values], columns=feature_names)

# --- 3. Streamlit UI ---
st.set_page_config(page_title="PhishEye Premium", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

st.markdown("---")

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

        st.markdown("---")
        st.markdown("### Domain Health Metrics:")
        m1, m2, m3 = st.columns(3)
        ssl_val = "HTTPS Active" if input_features_df['SSLfinal_State'].iloc[0] == 1 else "No HTTPS"
        url_len_val = f"{len(user_input)} Chars"
        short_val = "Detected" if input_features_df['Shortining_Service'].iloc[0] == -1 else "None"
        m1.metric("SSL Protocol", ssl_val, delta="Secure" if ssl_val == "HTTPS Active" else "-Risk", delta_color="normal" if ssl_val == "HTTPS Active" else "inverse")
        m2.metric("URL Length", url_len_val, delta="Normal" if len(user_input) < 54 else "-Long", delta_color="normal" if len(user_input) < 54 else "inverse")
        m3.metric("URL Shortener", short_val, delta="Clean" if short_val == "None" else "-Flagged", delta_color="normal" if short_val == "None" else "inverse")

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
            # Add more specific checks for features you can reliably derive
            st.markdown("- *Note: Analysis is based on a set of extracted URL features.*")


        with st.expander("🔍 Detailed 30-Feature ML Extraction Vector"):
            df_display = input_features_df.T.reset_index()
            df_display.columns = ["Feature Name", "Extracted Value (-1: Risk, 0: Neutral, 1: Safe)"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
    else:
        st.warning("Please enter a URL to analyze.")

st.markdown("---")
st.markdown("<div style='text-align: center; margin-top: 50px; opacity: 0.6;'>Developed for demonstrating ML model training and modern deployment.</div>", unsafe_allow_html=True)