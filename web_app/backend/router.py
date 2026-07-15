import base64
import io
import json
from logging import getLogger
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse

from lumen.preprocessing import preprocess_fused
from lumen.skin_tone import calculate_ita_subregions, get_fitzpatrick_label
from lumen.model_meta import encode_metadata, image_to_tensor, gradcam_fused

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
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    age: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    anatom_site: Optional[str] = Form(None),
):
    """Stream the fused metadata-aware analysis of an uploaded lesion photo.

    Metadata (age/sex/anatom_site) is optional; missing or unrecognised values fall
    back to the model's unknown/missing encoding, so an image-only request still works.
    """
    model = getattr(request.app.state, "ml_model", None)
    meta_cfg = getattr(request.app.state, "meta_cfg", None)

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "No file uploaded")

    def event_generator():
        if model is None or meta_cfg is None:
            yield format_sse(json.dumps({
                "step": "error",
                "message": "The analysis model is not loaded on the server.",
            }))
            return

        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            rgb = np.array(pil_image)  # RGB, HxWx3
        except Exception as e:
            yield format_sse(json.dumps({"step": "error", "message": f"Image decode failed: {e}"}))
            return

        # 1) Report the original image
        yield format_sse(json.dumps({
            "step": "load_image",
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "image_base64": base64.b64encode(contents).decode("utf-8"),
        }))

        # 2) Fused preprocessing: square crop -> 896 -> hair removal -> 448
        resize = meta_cfg["resize"]
        try:
            final_rgb, hair_mask, inpainted_rgb = preprocess_fused(rgb, target_size=resize)
        except Exception as e:
            yield format_sse(json.dumps({"step": "error", "message": f"Preprocessing failed: {e}"}))
            return

        _, mask_buf = cv2.imencode(".png", hair_mask)
        _, inp_buf = cv2.imencode(".png", cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR))
        yield format_sse(json.dumps({
            "step": "remove_hair",
            "hair_mask": base64.b64encode(mask_buf).decode(),
            "inpainted_image": base64.b64encode(inp_buf).decode(),
        }))

        # 3) Skin tone (ITA -> Fitzpatrick) + processed image.
        # calculate_ita_subregions consumes a BGR image (matches prior behaviour).
        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        avg_ita, _ = calculate_ita_subregions(final_bgr)
        skin_group = get_fitzpatrick_label(avg_ita)

        _, proc_buf = cv2.imencode(".png", final_bgr)
        yield format_sse(json.dumps({
            "step": "preprocess",
            "skin_group": skin_group,
            "processed_image": base64.b64encode(proc_buf).decode(),
        }))

        # 4) Tensor + metadata + fused prediction
        img_tensor = image_to_tensor(final_rgb)
        meta_tensor, meta_used = encode_metadata(age, sex, anatom_site, meta_cfg)

        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            prob = torch.sigmoid(logits).item()
            pred_cls = int(prob >= 0.5)

        yield format_sse(json.dumps({
            "step": "model_prediction",
            "probability": prob,
            "predicted_class": pred_cls,
            "metadata_used": meta_used,
        }))

        # 5) Grad-CAM over the TinyCNN branch
        try:
            cam = gradcam_fused(model, img_tensor, meta_tensor, out_size=resize)
            heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm_b64 = base64.b64encode(cv2.imencode(".png", heatmap)[1]).decode()
            yield format_sse(json.dumps({"step": "gradcam", "gradcam": hm_b64}))
        except Exception as e:
            logger.warning("Grad-CAM failed: %s", e)

        yield format_sse(json.dumps({"step": "done"}))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
