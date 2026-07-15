FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ /app/backend/
COPY frontend/dist/ /app/frontend/dist/

# We will serve the frontend from FastAPI by mounting it
# For now, just expose the API
EXPOSE 8000

# Default model path (overridden by docker-compose)
ENV MODEL_PATH=/app/models/adaptive_ipi_model

# Run Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
