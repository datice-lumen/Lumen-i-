# ────────────────────────────────
# Stage 1 – Build the frontend
# ────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install only the exact versions in package-lock.json
COPY frontend/package*.json ./
RUN npm ci --omit=dev

# Vite needs the source to build
COPY frontend ./
RUN npm run build          # outputs to /app/frontend/dist

# ────────────────────────────────
# Stage 2 – Install backend deps
# (kept separate so we can leverage
#  pip install layer caching)
# ────────────────────────────────
FROM python:3.11-slim AS python-deps

WORKDIR /wheels
COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel -r requirements.txt

# ────────────────────────────────
# Stage 3 – Final runtime image
# ────────────────────────────────
FROM python:3.11-slim

# A few system libs that Torch, Pillow… occasionally need
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# ── Python deps
COPY --from=python-deps /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

# ── Application code
COPY backend ./backend
COPY --from=frontend-build /app/frontend/dist ./frontend

EXPOSE 8000

# FastAPI entry-point; note $PORT from Render
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
