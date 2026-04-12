#!/usr/bin/env python3
"""Batch inference: preprocess images and output predictions to CSV.

Usage:
    python scripts/predict.py --images path/to/images --weights path/to/model.pth --output predictions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from lumen.model import CustomCNN, load_model
from lumen.inference import prepare_tensor, predict
from lumen.preprocessing import parallel_preprocess_for_inference


def main():
    parser = argparse.ArgumentParser(description="Batch melanoma prediction")
    parser.add_argument("--images", required=True, help="Folder with .jpg images")
    parser.add_argument("--weights", required=True, help="Path to model .pth weights")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel preprocessing")
    args = parser.parse_args()

    # Load model
    model = load_model(CustomCNN, args.weights)
    print(f"Model loaded from: {args.weights}")

    # Find images
    image_names = [f for f in os.listdir(args.images) if f.lower().endswith(".jpg")]
    print(f"Found {len(image_names)} images in '{args.images}'")

    # Preprocess
    num_processes = 1 if args.no_parallel else max(1, int(np.ceil(os.cpu_count() * 0.7)))
    print(f"Preprocessing with {num_processes} processes...")
    prep_results = parallel_preprocess_for_inference(image_names, args.images, num_processes)

    # Predict
    predictions = []
    for i, (image_name, preprocessed_img) in enumerate(prep_results):
        if preprocessed_img is None:
            continue
        tensor = prepare_tensor(preprocessed_img)
        prob, pred_class = predict(model, tensor, threshold=args.threshold)
        predictions.append((os.path.splitext(image_name)[0], pred_class))
        if (i + 1) % 500 == 0:
            print(f"  Predicted {i + 1} images...")

    # Save
    df = pd.DataFrame(predictions, columns=["image_name", "target"])
    df.to_csv(args.output, index=False)
    print(f"\n{len(predictions)} predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
