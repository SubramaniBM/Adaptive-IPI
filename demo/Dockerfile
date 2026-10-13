# ---------------------------------------------------------
# Stage 1: Build the React Frontend
# ---------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Install dependencies first for better caching
COPY frontend/package*.json ./
RUN npm ci

# Copy the rest of the frontend source and build
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------
# Stage 2: Serve Backend & Frontend via FastAPI
# ---------------------------------------------------------
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies if required for captum/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt captum

# Copy the backend code
COPY backend/ /app/backend/

# Embed the trained model weights directly into the image
COPY models/ /app/models/

# Copy the built frontend artifacts from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose the unified port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=/app/models/adaptive_ipi_model

# Run Uvicorn from the /app directory
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
