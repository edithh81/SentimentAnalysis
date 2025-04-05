FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir streamlit requests python-dotenv

# Copy application code
COPY ./app /app/app

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]