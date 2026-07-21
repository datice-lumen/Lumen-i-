import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from router import router as predict_router
from lumen.model_meta import load_dino, load_fused_model

logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).resolve().parent              # /app/backend
FRONTEND_DIST: Path = BASE_DIR.parent / "frontend"
# Two fused metadata-aware checkpoints (identical architecture), committed alongside
# the backend (~2.9 MB each). They share one frozen DINOv2 backbone at runtime.
#   phone: mobile fine-tune on phone close-ups (MILK10k), trained WITHOUT hair removal
#          -> fed the raw crop->448 image (router.py / preprocess_mobile).
#   derm:  original dermatoscopic model_10_6, trained WITH hair removal
#          -> fed the hair-removed 448 image (router.py / preprocess_fused).
MOBILE_MODEL_PATH: Path = BASE_DIR / "checkpoint_mobile_best.pt"
DERM_MODEL_PATH: Path = BASE_DIR / "najbolji_10_6.pt"


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
    """Load both fused models into memory once at startup, sharing one DINOv2 backbone.

    Each checkpoint loads independently: a failure leaves that mode unset (the endpoint
    then returns a clean error for that mode) rather than crashing the app or taking the
    other mode down with it. The shared DINOv2 is pulled from torch.hub on first run.
    """
    app.state.ml_models = {}
    app.state.meta_cfg = None

    try:
        dino = load_dino(device="cpu")
    except Exception as e:
        # No backbone -> no models. Leave state empty; endpoint returns a clean error.
        logger.exception("Failed to load DINOv2 backbone: %s", e)
        return

    for mode, path in (("phone", MOBILE_MODEL_PATH), ("derm", DERM_MODEL_PATH)):
        try:
            model, meta_cfg = load_fused_model(str(path), device="cpu", dino=dino)
            app.state.ml_models[mode] = model
            # Both checkpoints carry identical metadata config; keep the first loaded.
            if app.state.meta_cfg is None:
                app.state.meta_cfg = meta_cfg
            logger.info("Loaded %s model from %s", mode, path)
        except Exception as e:
            logger.exception("Failed to load %s model from %s: %s", mode, path, e)


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
