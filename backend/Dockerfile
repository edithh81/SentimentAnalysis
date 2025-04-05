FROM python:3.9-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
# Install specific compatible versions of PyTorch and torchtext
RUN pip install --no-cache-dir torch==2.0.0 torchtext==0.15.1

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app/app

# Copy models and data directories
# These will be mounted as volumes in docker-compose

# Set PYTHONPATH to resolve imports
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]