FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.6.0 psycopg2-binary boto3

# Create directories for artifacts and backing storage
RUN mkdir -p /mlflow/artifacts /mlflow/mlruns

# Set permissions to allow MLflow to write to directories
RUN chmod -R 777 /mlflow

# Expose port
EXPOSE 5000

# Set environment variables with defaults for better robustness
ENV BACKEND_STORE_URI=/mlflow/mlruns
ENV DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Start MLflow tracking server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT}