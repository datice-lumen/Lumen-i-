import os
import asyncio
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from router import router as predict_router
from model import download_model_from_gdrive, CustomEfficientNet


BASE_DIR: Path = Path(__file__).resolve().parent              # /app/backend
FRONTEND_DIST: Path = BASE_DIR.parent / "frontend"
MODEL_PATH: Path = BASE_DIR / "model_r0_75_r1_73_2904.pth"
GOOGLE_DRIVE_ID: str = "19SDsIq7dAEXQ7nq2MnFQxRjpQ7iqfBli"


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
# Startup: download weights (if absent) and load the model
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def load_model() -> None:
    """Fetch the model file from Google Drive (once) and load it into memory."""
    if not MODEL_PATH.exists():
        # non-blocking download executed in a thread so startup remains async
        await asyncio.to_thread(download_model_from_gdrive, MODEL_PATH, GOOGLE_DRIVE_ID)

    model = CustomEfficientNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    app.state.ml_model = model   # Later: request.app.state.ml_model


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
