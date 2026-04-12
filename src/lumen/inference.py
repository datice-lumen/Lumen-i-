"""Inference utilities: tensor preparation, prediction, and Grad-CAM."""

import numpy as np
import cv2
import torch


def prepare_tensor(img_np):
    """Convert a preprocessed numpy image to a batched model-ready tensor.

    Args:
        img_np: (H, W, 3) uint8 or float numpy array (RGB).

    Returns:
        torch.Tensor of shape (1, 3, H, W), float32, normalized to [0, 1].
    """
    assert img_np.shape[2] == 3, "Expected image with 3 channels (RGB)"
    img = img_np.astype(np.float32) / 255.0 if img_np.max() > 1.0 else img_np.astype(np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)


def predict(model, tensor, threshold=0.5):
    """Run a forward pass and return probability + binary prediction.

    Args:
        model: A model in eval mode that outputs raw logits.
        tensor: Batched input tensor (1, C, H, W).
        threshold: Classification threshold.

    Returns:
        (probability, predicted_class) — sigmoid probability and 0/1 int.
    """
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
        pred_class = int(prob >= threshold)
    return prob, pred_class


def apply_gradcam(model, input_tensor, target_layer):
    """Generate a Grad-CAM heatmap for the given input.

    Args:
        model: The CNN model (will be set to eval mode).
        input_tensor: (1, C, H, W) tensor. Must have requires_grad=True.
        target_layer: The convolutional layer to visualize (e.g. model.features[-1]).

    Returns:
        (cam, class_idx) — normalized [0, 1] CAM array resized to 224x224,
        and the predicted class index.
    """
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = int(torch.sigmoid(output).item() >= 0.5)
    score = output[:, 0]

    model.zero_grad()
    score.backward()

    grads_val = gradients[0][0].cpu().numpy()
    acts_val = activations[0][0].detach().cpu().numpy()

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts_val, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    handle_fwd.remove()
    handle_bwd.remove()

    return cam, class_idx
