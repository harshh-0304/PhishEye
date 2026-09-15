# 🛡️ PhishEye - Real-Time Phishing Detector

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

PhishEye is an intelligent web application that uses Machine Learning to analyze 30 key URL structural features in real-time to classify website URLs as **Legitimate** or **Phishing Attempts**.

---

## 🌟 Key Features

- **⚡ Real-Time Feature Extraction**: Analyzes SSL states, URL lengths, `@` symbols, IP hosting, domain hyphens, and shortener redirection.
- **📊 Confidence Score Gauge**: Provides exact percentage confidence level with visual progress bar indicators.
- **🎯 Preset Test Cases**: Instant sample buttons for fast testing (Phishing vs Safe vs Shortened URLs).
- **📜 Session Scan History**: Logs past URL scans during the active browser session for easy auditing.
- **🔍 30-Feature ML Inspector**: Interactive expander displaying complete 30-dimensional feature extraction vectors.
- **🎨 Glassmorphism UI**: Custom CSS dark theme styled with Geist and JetBrains Mono typography.

---

## 🛠️ Tech Stack & Model

- **Frontend/Dashboard**: Streamlit + Custom Dark Glassmorphism CSS
- **Model**: Scikit-Learn Random Forest Classifier (`phishing_detector_model.pkl`)
- **Language**: Python 3.x
- **Data Processing**: Pandas, Joblib, standard regex parsing

---

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r PhishEye/requirements.txt

# 2. Launch Streamlit App
streamlit run PhishEye/App.py
```
