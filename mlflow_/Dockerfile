FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.6.0 psycopg2-binary boto3

# Create directories for artifacts
RUN mkdir -p /mlflow/artifacts

# Expose port
EXPOSE 5000

# Start MLflow tracking server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT}