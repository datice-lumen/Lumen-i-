import base64
import json
from logging import getLogger

import cv2
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse

from img_process import check_resize, remove_hair, preprocess, prepare_tensor_for_model, get_fitzpatrick, apply_gradcam

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
    """
    Accepts an image upload, processes it step by step, and streams
    progress + results via SSE to the client.
    """
    model = request.app.state.ml_model  # <-- access it from app state

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No file uploaded")

    def event_generator():
        try:
            arr = np.frombuffer(contents, dtype=np.uint8)
            in_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if in_image is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            yield format_sse(json.dumps({"step": "error", "message": str(e)}))
            return

        yield format_sse(
            json.dumps({
                "step": "load_image",
                "height": in_image.shape[0],
                "width": in_image.shape[1],
                "image_base64": base64.b64encode(contents).decode("utf-8"),
            })
        )

        ok = check_resize(in_image)
        if not ok:
            in_image = cv2.resize(in_image, (500, 500), interpolation=cv2.INTER_LANCZOS4)

        hair_mask, inpainted = remove_hair(in_image)
        _, mask_buf = cv2.imencode(".png", hair_mask)
        _, inp_buf = cv2.imencode(".png", inpainted)
        yield format_sse(
            json.dumps({
                "step": "remove_hair",
                "hair_mask": base64.b64encode(mask_buf).decode("utf-8"),
                "inpainted_image": base64.b64encode(inp_buf).decode("utf-8"),
            })
        )

        preprocess_result = preprocess(in_image, TS=(224, 224), verbose=False)

        if preprocess_result is None:
            yield format_sse(json.dumps({
                "step": "error",
                "message": "Preprocessing failed (image not suitable)"
            }))
            return

        processed_img, metadata = preprocess_result

        skin_group = get_fitzpatrick(metadata["brightest_ITA"])

        _, processed_buf = cv2.imencode(".png", processed_img)
        processed_base64 = base64.b64encode(processed_buf).decode("utf-8")

        yield format_sse(json.dumps({
            "step": "preprocess",
            "skin_group": skin_group,
            "processed_image": processed_base64
        }))

        try:
            tensor = prepare_tensor_for_model(processed_img)
        except AssertionError as e:
            yield format_sse(json.dumps({
                "step": "error",
                "message": f"Tensor preparation failed: {str(e)}"
            }))
            return

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            pred_class = int(prob >= 0.5)

        yield format_sse(json.dumps({
            "step": "model_prediction",
            "probability": prob,
            "predicted_class": pred_class
        }))

        tensor.requires_grad = True
        cam, _ = apply_gradcam(model, tensor, model.features[-1])

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_b64 = base64.b64encode(cv2.imencode('.png', heatmap)[1]).decode("utf-8")

        yield format_sse(json.dumps({
            "step": "gradcam",
            "gradcam": heatmap_b64
        }))

        yield format_sse(json.dumps({"step": "done"}))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
