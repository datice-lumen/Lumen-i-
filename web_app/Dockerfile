###############################################################################
# Build context is the REPO ROOT (not web_app/), because the runtime image needs
# both the web_app/ sources and the top-level lumen package (pyproject.toml + src/).
# On Render this is configured via render.yaml (dockerContext: .) or, for a manually
# created service, Root Directory = repo root + Dockerfile Path = web_app/Dockerfile.
###############################################################################

###############################################################################
# Stage 1 ─ Build the Vue 3 + Naive-UI frontend
###############################################################################
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install all deps (dev + prod) so the Vite CLI is present, then build.
COPY web_app/frontend/package*.json ./
RUN npm ci

COPY web_app/frontend ./
RUN npm run build


###############################################################################
# Stage 2 ─ Python runtime
###############################################################################
FROM python:3.11-slim

# Minimal system libs required by Torch / Pillow / OpenCV at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Thread/allocator caps: Torch defaults to one OpenMP thread per *host* core (Render
# exposes far more than the plan's share), and glibc grows one malloc arena per
# thread — together this alone can push a 2 GB instance into OOM on the first request.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# ── 1) CPU-only Torch. The default PyPI `torch` bundles CUDA (~2.5 GB) which is
#       useless on Render's CPU instances and can exhaust the build's disk/time.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── 2) Runtime dependencies (curated web set; excludes ultralytics/SAM2, which the
#       web request path does not use). Torch is already satisfied above.
COPY web_app/backend/requirements.txt ./backend-requirements.txt
RUN pip install -r ./backend-requirements.txt

# ── 3) Install the lumen package for its code only (--no-deps: its pyproject pulls
#       ultralytics et al. that the web app doesn't need at runtime).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

# ── 4) Pre-bake DINOv2-S into the image so container cold-starts don't re-download
#       it from torch.hub over the network (Render's disk is ephemeral).
RUN python -c "from lumen.model_meta import load_dino; load_dino('cpu')"

# ── 5) Application code + built frontend.
COPY web_app/backend ./backend
COPY --from=frontend-build /app/frontend/dist ./frontend

# app.py does `from router import router`, so it must run from /app/backend.
WORKDIR /app/backend

# Render injects $PORT (defaults to 10000). `sh -c ... exec` expands the var AND
# replaces the shell with uvicorn so SIGTERM reaches it (graceful shutdown).
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
