FROM apache/airflow:2.7.1-python3.9

USER root

COPY requirements.txt .
# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --upgrade pip
# Install PyTorch with CPU only (smaller image) - using compatible versions
RUN pip install --no-cache-dir torch==1.13.1 torchtext==0.14.1

# Pin pydantic to version 1.x before installing other requirements
RUN pip install --no-cache-dir 'pydantic<2.0.0'

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add debugging tools
RUN pip install --no-cache-dir ipdb

# Set PYTHONPATH to include the DAGs directory
ENV PYTHONPATH=${PYTHONPATH}:/opt/airflow/dags