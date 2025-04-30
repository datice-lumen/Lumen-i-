# Stage 1: Build frontend
FROM node:18 AS frontend-builder
WORKDIR /app
COPY frontend ./frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Stage 2: Build backend + serve frontend
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend ./backend

# Copy frontend dist to backend's static path
COPY --from=frontend-builder /app/frontend/dist ./backend/static

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment vars (if needed)
# ENV MODEL_PATH=/app/backend/model/model_r0_75_r1_73_2904.pth

# Expose port
EXPOSE 8000

# Start app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
