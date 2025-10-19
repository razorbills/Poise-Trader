# 🐳 POISE TRADER - LEGENDARY CRYPTO BOT CONTAINER
# Multi-stage Docker build for optimal performance and security

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_enhanced.txt ./
RUN pip install --no-cache-dir --user -r requirements_enhanced.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRADING_ENVIRONMENT=production

# Create app user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/trading/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/config \
    && chown -R trading:trading /app

# Copy Python packages to user's local
USER trading
ENV PATH=/home/trading/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["python", "micro_trading_bot.py"]
