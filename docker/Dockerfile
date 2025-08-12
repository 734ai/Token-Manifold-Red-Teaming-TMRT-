FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p outputs findings logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TMRT_LOG_LEVEL=INFO
ENV TMRT_CACHE_DIR=/app/.cache

# Default command
CMD ["python", "-m", "tmrt.demo", "--help"]
