import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def check_resize(in_image):
    # check conditions for further preprocessing
    height, width = in_image.shape[:2]  # Get dimensions
    # if (width / height) != (3 / 2):
    #     return False
    if (width < 500 or height < 500):
        return False
    return True


def remove_hair(in_image):
    # Convert to grayscale
    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)

    # Apply black-hat filter (to detect dark hair-like structures)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a binary mask of hair
    _, hair_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

    # Final image, without hairs, inpainted
    inpainted_img = cv2.inpaint(in_image, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    return hair_mask, inpainted_img


def calculate_ITA_subregions(in_image, verbose):
    h, w = in_image.shape[:2]  # Get image dimensions
    size = int(0.2 * w)  # 20% of width/height

    # Define coordinates for each subregion as (x_start, y_start, x_end, y_end)
    regions = {
        "Top-Left": (0, 0, size, size),
        "Top-Right": (w - size, 0, w, size),
        "Bottom-Left": (0, h - size, size, h),
        "Bottom-Right": (w - size, h - size, w, h),
        "North": (w // 2 - size // 2, 0, w // 2 + size // 2, size),
        "South": (w // 2 - size // 2, h - size, w // 2 + size // 2, h),
        "West": (0, h // 2 - size // 2, size, h // 2 + size // 2),
        "East": (w - size, h // 2 - size // 2, w, h // 2 + size // 2),
    }

    if verbose:
        # Create a black mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Set white pixels for subregions
        for (x1, y1, x2, y2) in regions.values():
            mask[y1:y2, x1:x2] = 255

        mask_3channel = cv2.merge([mask, mask, mask])  # Make it (h, w, 3)

        # Apply mask on the original image
        masked_image = cv2.bitwise_and(in_image, mask_3channel)

        # Display the masked image
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        plt.title("Masked Image with Selected Subregions")
        plt.axis("off")
        plt.show()

    # get ITA for each subregion
    all8_ita_list = []
    avg_ita = 0

    for (x1, y1, x2, y2) in regions.values():
        sub_img = in_image[y1:y2, x1:x2]
        L, _, B = cv2.split(sub_img)
        L_mean = np.mean(L)
        B_mean = np.mean(B)
        ITA = np.arctan((L_mean - 50) / B_mean) * (180 / np.pi)
        # avg_ita += ITA
        all8_ita_list.append(ITA)

    avg_top2_ITA = sum(sorted(all8_ita_list, reverse=True)[:2]) / 2

    return avg_top2_ITA, all8_ita_list


def get_fitzpatrick(ita):
    if ita > 55:
        return "I (Very Light)"
    elif ita > 37:
        return "II (Light)"
    elif ita > 30:
        return "III (Intermediate)"
    elif ita > 14:
        return "IV (Tan)"
    elif ita > -30:
        return "V (Brown)"
    else:
        return "VI (Dark Brown/Black)"


def preprocess(orig_image, TS=(224, 224), verbose=False):
    h, w, _ = orig_image.shape

    # — square-crop center —
    new_w = h
    start_x = (w - new_w) // 2
    cropped_img = orig_image[:, start_x:start_x + new_w]
    if cropped_img.shape[0] != cropped_img.shape[1]:
        raise ValueError(
            f"Croppend image isn't perfect square! (h: {cropped_img.shape[0]}, w: {cropped_img.shape[1]})"
        )

    # — remove hair —
    hair_mask, crop_hairless_img = remove_hair(cropped_img)
    # Yield FIRST step:
    yield {
        "step": "remove_hair",
        "hair_mask": hair_mask,
        "inpainted": crop_hairless_img
    }

    # — resize to target TS and compute ITA & Fitzpatrick —
    small_crop_hairless_img = cv2.resize(
        crop_hairless_img, TS, interpolation=cv2.INTER_LANCZOS4
    )
    avg_top2_ITA, all8_ita_list = calculate_ITA_subregions(
        small_crop_hairless_img, verbose
    )
    skin_group = get_fitzpatrick(avg_top2_ITA)

    # Yield SECOND step:
    yield {
        "step": "preprocess",
        "processed_img": small_crop_hairless_img,
        "brightest_ITA": avg_top2_ITA,
        "ita_list": all8_ita_list,
        "skin_group": skin_group
    }


def prepare_tensor_for_model(img_np):
    # Check if shape is (224, 224, 3)
    assert img_np.shape[2] == 3, "Expected image with 3 channels (RGB)"

    print("MAX value before", np.max(img_np))
    img_np = img_np.astype(np.float32) / 255.0
    print("MAX value after", np.max(img_np))

    # Convert to tensor and reshape to [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, 224, 224]

    # Normalize (ImageNet stats for EfficientNet)
    # imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # img_tensor = (img_tensor - imagenet_mean) / imagenet_std

    # Add batch dimension: [1, 3, 224, 224]
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def apply_gradcam(model, input_tensor, target_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    class_idx = int(torch.sigmoid(output).item() >= 0.5)
    score = output[:, 0]

    # Backward
    model.zero_grad()
    score.backward()

    # Get data
    grads_val = gradients[0][0].cpu().numpy()  # shape: C x H x W
    acts_val = activations[0][0].detach().cpu().numpy()  # shape: C x H x W

    weights = np.mean(grads_val, axis=(1, 2))  # GAP over H and W
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts_val, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    # Clean up
    handle_fwd.remove()
    handle_bwd.remove()
    return cam, class_idx
