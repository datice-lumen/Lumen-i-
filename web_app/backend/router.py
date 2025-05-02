import base64
import cv2
import json
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from logging import getLogger

from img_process import preprocess, prepare_tensor_for_model, apply_gradcam

logger = getLogger(__name__)

router = APIRouter(
    prefix="/image",
    tags=["image"],
)


def format_sse(data: str, event: str = None) -> str:
    """
    Simple formatter for Server-Sent Events:
      event: <event-name>    (optional)
      data: <json-payload>
    """
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += f"data: {data}\n\n"
    return msg


@router.post("/process")
async def process_image(request: Request, file: UploadFile = File(...)):
    model = request.app.state.ml_model
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "No file uploaded")

    def event_generator():
        try:
            arr = np.frombuffer(contents, dtype=np.uint8)
            in_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if in_image is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            yield format_sse(json.dumps({"step": "error", "message": str(e)}))
            return

        # tell client about dims & original upload
        yield format_sse(json.dumps({
            "step": "load_image",
            "height": in_image.shape[0],
            "width": in_image.shape[1],
            "image_base64": base64.b64encode(contents).decode("utf-8"),
        }))

        # --- 2) single call to preprocess, which yields TWO events ---
        processed_img = None
        for ev in preprocess(in_image, TS=(224, 224), verbose=False):
            if ev["step"] == "remove_hair":
                # encode mask + inpainted
                _, mask_buf = cv2.imencode(".png", ev["hair_mask"])
                _, inp_buf = cv2.imencode(".png", ev["inpainted"])
                yield format_sse(json.dumps({
                    "step": "remove_hair",
                    "hair_mask": base64.b64encode(mask_buf).decode(),
                    "inpainted_image": base64.b64encode(inp_buf).decode(),
                }))
            elif ev["step"] == "preprocess":
                # encode final TS×TS
                _, proc_buf = cv2.imencode(".png", ev["processed_img"])
                processed_img = ev["processed_img"]
                yield format_sse(json.dumps({
                    "step": "preprocess",
                    "skin_group": ev["skin_group"],
                    "processed_image": base64.b64encode(proc_buf).decode(),
                }))

        # --- 3) tensor prep & model prediction ---
        try:
            tensor = prepare_tensor_for_model(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        except AssertionError as e:
            yield format_sse(json.dumps({
                "step": "error",
                "message": f"Tensor preparation failed: {e}"
            }))
            return

        with torch.no_grad():
            out = model(tensor)
            prob = torch.sigmoid(out).item()
            pred_cls = int(prob >= 0.5)

        yield format_sse(json.dumps({
            "step": "model_prediction",
            "probability": prob,
            "predicted_class": pred_cls
        }))

        # --- 4) Grad-CAM ---
        tensor.requires_grad_(True)
        cam, _ = apply_gradcam(model, tensor, model.features[-1])
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        hm_b64 = base64.b64encode(cv2.imencode('.png', heatmap)[1]).decode()

        yield format_sse(json.dumps({
            "step": "gradcam",
            "gradcam": hm_b64
        }))

        yield format_sse(json.dumps({"step": "done"}))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
