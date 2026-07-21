import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from router import router as predict_router
from lumen.model_meta import load_fused_model

logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).resolve().parent              # /app/backend
FRONTEND_DIST: Path = BASE_DIR.parent / "frontend"
# Fused metadata-aware model, mobile fine-tuned, committed alongside the backend (2.9 MB).
# Fine-tuned on phone close-ups (MILK10k mobile); trained WITHOUT hair removal, so the
# request path feeds it the raw crop->448 image (see router.py / preprocess_mobile).
MODEL_PATH: Path = BASE_DIR / "checkpoint_mobile_best.pt"


app = FastAPI(title="FastAPI-Vue monorepo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# REST/ML routes
app.include_router(predict_router)

# Serve the single-page application
# html=True makes every unknown path fall back to index.html (Vue router history mode)
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")


# --------------------------------------------------------------------------- #
# Startup: build the fused model and load committed weights.
# DINOv2-S is pulled from torch.hub on first run (needs network, then cached).
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def load_model() -> None:
    """Load the fused metadata-aware model into memory once at startup."""
    try:
        model, meta_cfg = load_fused_model(str(MODEL_PATH), device="cpu")
        app.state.ml_model = model
        app.state.meta_cfg = meta_cfg
        logger.info("Loaded fused model from %s", MODEL_PATH)
    except Exception as e:
        # Leave state unset so the endpoint returns a clean error rather than crashing
        # the whole app (e.g. DINOv2 hub download unavailable, missing checkpoint).
        app.state.ml_model = None
        app.state.meta_cfg = None
        logger.exception("Failed to load model: %s", e)


# --------------------------------------------------------------------------- #
# Optional explicit root (StaticFiles already handles it, but handy for local tests)
# --------------------------------------------------------------------------- #

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    return FileResponse(FRONTEND_DIST / "index.html")


# --------------------------------------------------------------------------- #
# Local development entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    port: int = int(os.environ.get("PORT", 8000))
    reload: bool = bool(os.environ.get("DEV"))  # set DEV=1 for autoreload locally

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
