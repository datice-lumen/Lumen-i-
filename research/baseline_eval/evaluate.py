# This script works on raw data, does the preprocessing + inference on model

import os
import multiprocessing
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn


# INPUT PARAMETERS
# /////////////////////////////////////
PARALLEL = True   # set True if you want to preprocess images in parallel
MODEL_WEIGHTS = [
    "/home/datice/data/models_448/model_fold_0.pth",
    "/home/datice/data/models_448/model_fold_1.pth",
    "/home/datice/data/models_448/model_fold_2.pth",
    "/home/datice/data/models_448/model_fold_3.pth",
]

INPUT_FOLDER = "/home/datice/data/baseline_eval/all_imgs"
OUTPUT_CSV = "predictions_448_ensemble.csv"
CHUNK_SIZE = 5000   # work in chunks of this size, avoids heavy RAM usage
TS = (448, 448)     # target image size


# MODEL INITIALIZATION
# /////////////////////////////////////
class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# HELPER FUNCTIONS
# /////////////////////////////////////
def resolve_input_folder(input_arg):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    relative_path = os.path.join(parent_dir, input_arg)

    if os.path.exists(relative_path):
        print(f"Found input folder at: {relative_path}")
        return relative_path
    elif os.path.exists(input_arg):
        print(f"Found input folder at: {input_arg}")
        return input_arg
    else:
        raise FileNotFoundError(f"Input folder '{input_arg}' not found in parent directory or as absolute path.")


# FUNCTIONS FOR PREPROCESSING
# /////////////////////////////////////
def remove_hair(in_image, kernel_size=25):
    gray = cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

    inpainted_img = cv2.inpaint(in_image, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    return hair_mask, inpainted_img


def preprocess(image_path, target_size=TS):
    # LOAD IMAGE
    if os.path.exists(image_path):
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            print(f"Failed to read image: {image_path}")
            return None
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Image not found: {image_path}")
        return None

    # CENTRAL SQUARE CROP
    h, w, _ = orig_image.shape

    side = min(h, w)
    start_y = (h - side) // 2
    start_x = (w - side) // 2

    cropped_img = orig_image[start_y:start_y + side, start_x:start_x + side]

    if cropped_img.shape[0] != cropped_img.shape[1]:
        print(f"Cropped image isnt perfect square! (h: {cropped_img.shape[0]}, w: {cropped_img.shape[1]})")
        return None

    # RESIZE FOR HAIR REMOVAL: 2x target size, min 800, with proportional kernel
    intermed = max(800, target_size[0] * 2)
    raw_kernel = round(25 * intermed / 800)
    hair_kernel = raw_kernel if raw_kernel % 2 == 1 else raw_kernel + 1
    cropped_img = cv2.resize(cropped_img, (intermed, intermed), interpolation=cv2.INTER_AREA)

    # REMOVE HAIR
    hair_mask, crop_hairless_img = remove_hair(cropped_img, kernel_size=hair_kernel)

    # RESIZE TO TARGET SIZE
    small_crop_hairless_img = cv2.resize(crop_hairless_img, target_size, interpolation=cv2.INTER_LANCZOS4)

    return prepare_tensor_for_model(small_crop_hairless_img)


def prepare_tensor_for_model(img_np):
    assert img_np.shape[2] == 3, "Expected image with 3 channels (RGB)"

    img_np = img_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


# FUNCTION FOR PARALLEL PREPROCESSING
# /////////////////////////////////////
def preprocess_with_path(full_path):
    image_name = os.path.basename(full_path)
    processed = preprocess(full_path)
    return (image_name, processed)


def parallel_preprocess(images_name, num_processes, where_folder):
    """Run image preprocessing in parallel and return (image_name, processed_image) tuples."""
    inputs = [os.path.join(where_folder, image_name) for image_name in images_name]

    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap(preprocess_with_path, inputs):
            results.append(result)

    return results


# MAIN
# /////////////////////////////////////
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # GPU CHECK
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available. Exiting.")
        raise SystemExit(1)

    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print(f"Target size: {TS}")

    input_folder = resolve_input_folder(INPUT_FOLDER)
    output_csv = OUTPUT_CSV

    # LOADING ALL MODELS (ensemble)
    models = []
    for model_path in MODEL_WEIGHTS:
        m = CustomEfficientNet()
        try:
            m.load_state_dict(torch.load(model_path, map_location=device))
            m = m.to(device)
            m.eval()
            models.append(m)
            print(f"Loaded: {model_path}")
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            raise SystemExit(1)

    print(f"\nEnsemble of {len(models)} models.")

    images_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    print(f"Found {len(images_to_process)} image files in '{input_folder}'.")

    if PARALLEL is True:
        num_processes = int(np.ceil(multiprocessing.cpu_count() * 0.7))
    else:
        num_processes = 1

    print("\nNum. processes:", num_processes)
    print(f"Chunk size: {CHUNK_SIZE}")
    print()

    predictions = []
    failed_preprocessing = []
    successful_inference = 0
    total_images = len(images_to_process)
    processed_total = 0

    for chunk_start in range(0, total_images, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
        chunk_images = images_to_process[chunk_start:chunk_end]

        prep_results = parallel_preprocess(chunk_images, num_processes, where_folder=input_folder)

        for image_name, preprocessed_image_tensor in prep_results:
            if preprocessed_image_tensor is None:
                failed_preprocessing.append(image_name)
                continue

            preprocessed_image_tensor = preprocessed_image_tensor.to(device)

            # ensemble: average sigmoid probabilities across all models
            probs = []
            with torch.no_grad():
                for m in models:
                    pred = m(preprocessed_image_tensor)
                    probs.append(torch.sigmoid(pred).item())
            prob = float(np.mean(probs))
            binary_pred = int(prob > 0.5)

            predictions.append((
                os.path.splitext(image_name)[0],  # image_name without extension
                binary_pred,
                prob
            ))
            successful_inference += 1

        processed_total = chunk_end
        percent = (processed_total / total_images) * 100
        print(f"\nProgress: {processed_total}/{total_images} ({percent:.2f}%)\n")

        del prep_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_csv_path = os.path.join(script_dir, output_csv)
    df = pd.DataFrame(predictions, columns=["image_name", "target", "conf"])
    df.to_csv(output_csv_path, index=False)

    print("\n==================== SUMMARY ====================")
    print(f"Images found in folder:       {len(images_to_process)}")
    print(f"Successful model inference:   {successful_inference}")
    print(f"Failed preprocessing:         {len(failed_preprocessing)}")
    print(f"CSV rows written:             {len(df)}")
    print(f"Output CSV:                   {output_csv_path}")

    if len(images_to_process) == successful_inference:
        print("All images from the folder successfully passed through the model.")
    else:
        print("Not all images from the folder passed through the model.")

        if failed_preprocessing:
            print("\nImages that failed preprocessing:")
            for img in failed_preprocessing[:20]:
                print(f" - {img}")

            if len(failed_preprocessing) > 20:
                print(f" ... and {len(failed_preprocessing) - 20} more.")