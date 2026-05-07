import base64
import cv2
import io
import json
import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from logging import getLogger

from lumen.preprocessing import remove_hair, square_crop
from lumen.skin_tone import calculate_ita_subregions, get_fitzpatrick_label
from lumen.inference import prepare_tensor, apply_gradcam

logger = getLogger(__name__)

router = APIRouter(
    prefix="/image",
    tags=["image"],
)


def format_sse(data: str, event: str = None) -> str:
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
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            in_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            yield format_sse(json.dumps({"step": "error", "message": f"Image decode failed: {e}"}))
            return

        # 1) Report original image
        yield format_sse(json.dumps({
            "step": "load_image",
            "height": in_image.shape[0],
            "width": in_image.shape[1],
            "image_base64": base64.b64encode(contents).decode("utf-8"),
        }))

        # 2) Square crop + hair removal
        cropped = square_crop(in_image)
        hair_mask, inpainted = remove_hair(cropped)

        _, mask_buf = cv2.imencode(".png", hair_mask)
        _, inp_buf = cv2.imencode(".png", inpainted)
        yield format_sse(json.dumps({
            "step": "remove_hair",
            "hair_mask": base64.b64encode(mask_buf).decode(),
            "inpainted_image": base64.b64encode(inp_buf).decode(),
        }))

        # 3) Resize + ITA / skin group
        TS = (224, 224)
        processed_img = cv2.resize(inpainted, TS, interpolation=cv2.INTER_LANCZOS4)
        avg_ita, _ = calculate_ita_subregions(processed_img)
        skin_group = get_fitzpatrick_label(avg_ita)

        _, proc_buf = cv2.imencode(".png", processed_img)
        yield format_sse(json.dumps({
            "step": "preprocess",
            "skin_group": skin_group,
            "processed_image": base64.b64encode(proc_buf).decode(),
        }))

        # 4) Tensor prep + model prediction
        try:
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            tensor = prepare_tensor(rgb_img)
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
            "predicted_class": pred_cls,
        }))

        # 5) Grad-CAM
        tensor.requires_grad_(True)
        cam, _ = apply_gradcam(model, tensor, model.features[-1])
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        hm_b64 = base64.b64encode(cv2.imencode(".png", heatmap)[1]).decode()

        yield format_sse(json.dumps({
            "step": "gradcam",
            "gradcam": hm_b64,
        }))

        yield format_sse(json.dumps({"step": "done"}))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
