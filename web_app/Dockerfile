###############################################################################
# Stage 1 ─ Build the Vue 3 + Naive-UI frontend
###############################################################################
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# 1) Install all deps (dev + prod) so the Vite CLI is present
COPY frontend/package*.json ./
RUN npm ci

# 2) Copy source and run the Vite build
COPY frontend ./
RUN npm run build          

# 3) (optional) Drop dev-deps to keep this layer lean
RUN npm prune --production



###############################################################################
# Stage 2 ─ Pre-build Python wheels (speeds up the final image build)
###############################################################################
FROM python:3.11-slim AS python-deps

WORKDIR /wheels
COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel -r requirements.txt



###############################################################################
# Stage 3 ─ Final runtime image
###############################################################################
FROM python:3.11-slim

# Minimal system libs sometimes required by Torch, Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Helpful Python env flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PORT=8000                

WORKDIR /app

# ── Python dependencies
COPY --from=python-deps /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

# ── Application code
COPY backend ./backend                   
COPY --from=frontend-build /app/frontend/dist ./frontend

EXPOSE 8000

# Render starts the container with `docker run -p $PORT:$PORT …`
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
