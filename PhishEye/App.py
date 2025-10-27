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
st.set_page_config(page_title="PhishEye Detector", page_icon="👁️", layout="wide")

st.title("👁️ PhishEye: Real-time Phishing Detector")
st.markdown("---")
st.markdown("""
    This application helps detect potential phishing attempts by analyzing various features of a URL.
    Input a URL below to get an instant classification.
""")

user_input = st.text_input("Enter URL to analyze:", "https://security-update-bankofamerica.com/login?id=123", help="Try entering a suspicious URL or a legitimate one like https://google.com")

st.markdown("---")

if st.button("Analyze URL", type="primary"):
    if user_input:
        st.info("Analyzing the provided URL...")
        
        # Get features from the input URL
        input_features_df = get_features_from_url(user_input)

        # Make prediction
        prediction = model.predict(input_features_df)[0]
        prediction_proba = model.predict_proba(input_features_df)[0]

        st.markdown("### Analysis Result:")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 0: # 0 for phishing
                st.error("🚨 PHISHING DETECTED!")
            else: # 1 for safe
                st.success("✅ SAFE - No Phishing Detected")

        with col2:
            confidence = prediction_proba[prediction] * 100
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.markdown("---")
        st.markdown("### Reasoning & Key Indicators:")

        # Provide reasoning based on the input features and their importance
        if prediction == 0: # If classified as phishing
            st.warning("This URL exhibits characteristics commonly associated with phishing.")
            
            # Highlight specific features that contributed to the phishing detection
            if input_features_df['SSLfinal_State'].iloc[0] == -1:
                st.markdown("- **No HTTPS (SSLfinal_State: -1):** The connection is not secure, a major red flag for phishing.")
            if input_features_df['having_At_Symbol'].iloc[0] == -1:
                st.markdown("- **Uses '@' Symbol (having_At_Symbol: -1):** This can be used to deceive users about the true domain.")
            if input_features_df['URLURL_Length'].iloc[0] == -1:
                st.markdown(f"- **Very Long URL (URLURL_Length: -1):** Long URLs ({len(user_input)} characters) often hide malicious destinations.")
            if input_features_df['Shortining_Service'].iloc[0] == -1:
                st.markdown("- **Uses URL Shortener (Shortining_Service: -1):** Obscures the true destination, a common phishing tactic.")
            if input_features_df['having_Sub_Domain'].iloc[0] == -1 or input_features_df['having_Sub_Domain'].iloc[0] == 0:
                st.markdown("- **Multiple Subdomains (having_Sub_Domain: -1/0):** Can mimic legitimate sites while redirecting to malicious ones.")
            if input_features_df['Prefix_Suffix'].iloc[0] == -1:
                st.markdown("- **Hyphenated Domain (Prefix_Suffix: -1):** Often used to make domains look similar to legitimate ones.")
            # Add more specific checks for features you can reliably derive
            
            st.markdown("- *Note: Some features were approximated due to the complexity of real-time extraction.*")
            
        else: # If classified as safe
            st.info("The URL appears to be legitimate based on the analysis.")
            if input_features_df['SSLfinal_State'].iloc[0] == 1:
                st.markdown("- **Uses HTTPS (SSLfinal_State: 1):** Indicates a secure connection.")
            if input_features_df['having_At_Symbol'].iloc[0] == 1:
                st.markdown("- **No '@' Symbol (having_At_Symbol: 1):** The URL does not contain the deceptive '@' symbol.")
            if input_features_df['URLURL_Length'].iloc[0] == 1 or input_features_df['URLURL_Length'].iloc[0] == 0:
                st.markdown(f"- **Normal URL Length (URLURL_Length: {input_features_df['URLURL_Length'].iloc[0]}):** The URL length ({len(user_input)} characters) falls within typical ranges.")
            # Add more specific checks for features you can reliably derive
            st.markdown("- *Note: Analysis is based on a set of extracted URL features.*")


        st.expander("Show All Extracted Features (for debugging/detail)").dataframe(input_features_df.T) # Transpose for better viewing
        
    else:
        st.warning("Please enter a URL to analyze.")

st.markdown("---")
st.markdown("Developed for demonstrating ML model training, feature engineering, and deployment.")