FROM python:3.11-slim

# Prevent python from writing pyc files to disc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/PhishEye:/app

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY PhishEye/requirements.txt /app/PhishEye/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/PhishEye/requirements.txt

# Copy application files
COPY PhishEye/ /app/PhishEye/
COPY README.md /app/README.md

# Expose Streamlit dashboard and API ports
EXPOSE 8501
EXPOSE 8000

# Healthcheck for container status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command launches Streamlit Cyber Guard dashboard
CMD ["streamlit", "run", "PhishEye/App.py", "--server.port=8501", "--server.address=0.0.0.0"]
