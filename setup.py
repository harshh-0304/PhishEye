from setuptools import setup, find_packages

setup(
    name="phisheye",
    version="2.1.0",
    description="Real-Time Machine Learning Phishing URL & Threat Detection Engine",
    author="harshh-0304",
    author_email="bakasakuta33@gmail.com",
    url="https://github.com/harshh-0304/PhishEye",
    packages=find_packages(where="PhishEye"),
    package_dir={"": "PhishEye"},
    install_requires=[
        "joblib>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.30.0",
    ],
    entry_points={
        "console_scripts": [
            "phisheye=phisheye.cli:main",
        ],
    },
    python_requires=">=3.10",
    license="MIT",
)
