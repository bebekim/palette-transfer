# ABOUTME: Dockerfile for local testing that mimics Railway's Railpack environment
# ABOUTME: Uses Python 3.10 with uv package manager like Railway

FROM python:3.10-slim

# Install system dependencies for dlib/face-recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_ENV=production

# Run migrations then start gunicorn
# --preload runs create_app() once before forking workers, avoiding race conditions in db.create_all()
CMD flask db upgrade && gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 --preload "app:create_app()"
