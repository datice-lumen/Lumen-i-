import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router as predict_router
from model import download_model_from_gdrive, CustomEfficientNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(predict_router)

MODEL_PATH = "model_r0_75_r1_73_2904.pth"
GOOGLE_DRIVE_ID = "19SDsIq7dAEXQ7nq2MnFQxRjpQ7iqfBli"


@app.on_event("startup")
def load_model():
    model_path = "model_r0_75_r1_73_2904.pth"
    model_id = "19SDsIq7dAEXQ7nq2MnFQxRjpQ7iqfBli"
    download_model_from_gdrive(model_path, model_id)

    model = CustomEfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    app.state.ml_model = model


@app.get("/")
def read_root():
    return {"message": "Melanoma Detection API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
