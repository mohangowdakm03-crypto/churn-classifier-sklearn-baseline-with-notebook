# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

# Install system deps for LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure data directory exists and is owned by app user
RUN mkdir -p data && chown -R app:app /app

# Switch to non-root user
USER app

# Expose only the API port (Streamlit removed)
EXPOSE 8000

# Default: run the FastAPI server (no --reload in production)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
