# ====================================================================
# Adaptive-IPI v2: Production Universal Docker Container
# Cross-platform CPU container for Linux, macOS, and Windows
# ====================================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-optimized wheel and python dependencies
COPY v2/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and model weights
COPY v2/ /app/v2/
COPY demo/models/ /app/demo/models/

# Expose server port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "v2.demo.server:app", "--host", "0.0.0.0", "--port", "8080"]
