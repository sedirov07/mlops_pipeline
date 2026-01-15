FROM apache/airflow:2.8.4-python3.11

# Install system dependencies required by PyCaret/ML libraries
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install API dependencies
RUN pip install --no-cache-dir uvicorn fastapi pandas joblib python-dotenv