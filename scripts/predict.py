#!/usr/bin/env python3
"""Batch inference: preprocess images and output predictions to CSV.

Usage:
    python scripts/predict.py --images path/to/images --weights path/to/model.pth --output predictions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from lumen.model import CustomCNN, load_model
from lumen.inference import prepare_tensor, predict
from lumen.preprocessing import parallel_preprocess_for_inference


def main():
    parser = argparse.ArgumentParser(description="Batch melanoma prediction")
    parser.add_argument("--images", required=True, help="Folder with .jpg images")
    parser.add_argument("--weights", required=True, help="Path to model .pth weights")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Process images in chunks of this size to bound memory (default: 5000)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel preprocessing")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    print(f"Using device: {device}")

    # Load model
    model = load_model(CustomCNN, args.weights, device=device)
    print(f"Model loaded from: {args.weights}")

    # Find images
    image_names = [f for f in os.listdir(args.images) if f.lower().endswith(".jpg")]
    print(f"Found {len(image_names)} images in '{args.images}'")

    # Preprocess + predict in chunks (bounds peak RAM/VRAM)
    num_processes = 1 if args.no_parallel else max(1, int(np.ceil(os.cpu_count() * 0.7)))
    print(f"Preprocessing with {num_processes} processes, chunk size {args.chunk_size}")

    predictions = []
    failed = 0
    total = len(image_names)
    for start in range(0, total, args.chunk_size):
        end = min(start + args.chunk_size, total)
        chunk_names = image_names[start:end]

        prep_results = parallel_preprocess_for_inference(chunk_names, args.images, num_processes)

        for image_name, preprocessed_img in prep_results:
            if preprocessed_img is None:
                failed += 1
                continue
            tensor = prepare_tensor(preprocessed_img).to(device)
            prob, pred_class = predict(model, tensor, threshold=args.threshold)
            predictions.append((os.path.splitext(image_name)[0], pred_class, prob))

        del prep_results
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"  Progress: {end}/{total} ({end / total * 100:.1f}%)")

    # Save
    df = pd.DataFrame(predictions, columns=["image_name", "target", "conf"])
    df.to_csv(args.output, index=False)
    print(f"\n{len(predictions)} predictions saved to: {args.output}")
    if failed:
        print(f"Failed preprocessing: {failed}")


if __name__ == "__main__":
    main()
