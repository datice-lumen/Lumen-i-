import os
import sys

import multiprocessing
import pandas as pd
import numpy as np
import cv2
import torch #2.6.0+cu124 bi trebao (ja imam 2.6.0 i dela sve)
import torch.nn as nn
#print(torch.__version__)








# (OPTIONAL) (ADDITIONAL) INPUT PARAMETERS
#/////////////////////////////////////
PARALLEL = True   # set True if you want to preprocess images in parallel with aprox. 70% of cores available (speeds up significantly)
MODEL_WEIGHTS = "MODEL_58_84_0205.pth"



# MODEL INITIALIZATION
#/////////////////////////////////////
# this is only models structure, pretrained weights will be added into it

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
            nn.Linear(128*14*14, 256),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 1)   # Binary classification (logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# HELPER FUNCTIONS
#/////////////////////////////////////

def resolve_input_folder(input_arg):
    # This functions checks if folder exists inside parent folder, or it is absolute path to somewhere else
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
#/////////////////////////////////////


def check_resize(in_image):
    # check conditions for further preprocessing
    height, width = in_image.shape[:2] 
    if (width / height ) != (3/2):
      return False
    elif (width < 500 or height < 500):
      return False
    else:
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




def preprocess(image_path, TS = (224, 224)):

    # LOAD IMAGE
    if os.path.exists(image_path):
        orig_image = cv2.imread(image_path)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    else:
        print(f"Image not found: {image_path}")
        return None

    orig_image = cv2.resize(orig_image, (1200, 800), interpolation=cv2.INTER_AREA) 
    # resizing with 2 steps, to reduce computation in-between steps, 
    # and to preserve as much quality for hair removal

    if (check_resize(orig_image) == False ):
        print("Input image isnt either low quality or is not in 1.5 ratio!")
        return None

    # SQUARE RESIZE
    # we will remove portions from the left and right of the image, because they sometimes include dermatoscope in it, wich can be missleading for
    # Fitzpatrick calculations. Also, this way we get square (1:1 ratio) which is great for data augmentation in the future if needed
    h, w, _ = orig_image.shape  # Get height and width

    new_w = h
    start_x = (w - new_w) // 2
    cropped_img = orig_image[:, start_x:start_x + new_w]
    if (cropped_img.shape[0] != cropped_img.shape[1]):
        print(f"Croppend image isnt perfect square! (h: {cropped_img.shape[0]}, w: {cropped_img.shape[1]})")
        return None


    # REMOVE HAIR
    hair_mask, crop_hairless_img = remove_hair(cropped_img)

    # RESIZE TO TARGET SIZE
    small_crop_hairless_img = cv2.resize(crop_hairless_img, TS, interpolation=cv2.INTER_LANCZOS4)

    return prepare_tensor_for_model(small_crop_hairless_img) 
    # returns tensor, ready for model interference


def prepare_tensor_for_model(img_np):
    # Check if shape is (224, 224, 3)
    assert img_np.shape[2] == 3, "Expected image with 3 channels (RGB)"

    #print("MAX value before", np.max(img_np))
    img_np = img_np.astype(np.float32) / 255.0
    #print("MAX value after", np.max(img_np))

    # Convert to tensor and reshape to [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, 224, 224]
    

    # Add batch dimension: [1, 3, 224, 224]
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# FUNCTION FOR PARALLEL PREPROCESSING
#/////////////////////////////////////
def preprocess_with_path(full_path):
    image_name = os.path.basename(full_path)
    processed = preprocess(full_path)
    return (image_name, processed)

def parallel_preprocess(images_name, num_processes, where_folder):
    """Run image preprocessing in parallel and return (image_name, processed_image) tuples."""

    pool = multiprocessing.Pool(processes=num_processes)

    inputs = [os.path.join(where_folder, image_name) for image_name in images_name]

    #track progress
    results = []
    for idx, result in enumerate(pool.imap(preprocess_with_path, inputs)):
        results.append(result)
        
        if (idx + 1) % 500 == 0:
            print(f"\t-> Preprocessed {idx + 1} images...")

    pool.close()
    pool.join()

    return results
    # [("img001.jpg", processed_img1), ("img002.jpg", processed_img2), ...]







# MAIN
#/////////////////////////////////////

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 predict.py <INPUT_FOLDER> <OUTPUT_CSV>")
        print("<INPUT_FOLDER> - can be just folder name (if it is inside DATICE_lumen_ds_2025) or absolute path if it is elsewhere")
        print("<OUTPUT_CSV> - this file will be generated after execution inside DATICE_lumen_ds_2025")
        sys.exit(1)

    input_folder = resolve_input_folder(sys.argv[1])
    output_csv = sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))




    # LOADING MODEL
    model = CustomEfficientNet() 
    model_path = os.path.join(script_dir, MODEL_WEIGHTS)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model weights loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Failed to load model weights from: {model_path}")
        print(f"Error: {e}")
        sys.exit(1)

    model.eval() 
    # now we have pretrained weights loaded and our model is ready to use


    
    images_to_process= [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    #images_to_process = images_to_process[1000:1200] ########################################################################################DELETE

    # Count and display
    print(f"Found {len(images_to_process)} image files in '{input_folder}'.")





    if PARALLEL == True:
        num_processes = int(np.ceil(multiprocessing.cpu_count() * 0.7)) # Use around 70% of core available
    else:
        num_processes = 1

    print("\nNum. processes:", num_processes)
    print()

    prep_results = parallel_preprocess(images_to_process, num_processes, where_folder=input_folder)
    print("\nImages preprocessed:", len(prep_results))
    print()

    # create empty df with columns ['image_name', 'target']
    predictions = []
    cnt = 0

    for image_name, preprocessed_image_tensor in prep_results:
        cnt += 1

        if cnt % 500 == 0:
            print(f"\t-> Predicted {cnt} images...")


        # run interface on model
        with torch.no_grad():
            pred = model(preprocessed_image_tensor) 

        prob = torch.sigmoid(pred).item()
        binary_pred = int(prob > 0.5)  # convert probability to binary class

        # save its result under target in df, with image name
        predictions.append((os.path.splitext(image_name)[0], binary_pred)) # REMOVE extension from image_name
    


    output_csv_path = os.path.join(script_dir, output_csv)
    df = pd.DataFrame(predictions, columns=["image_name", "target"])
    df.to_csv(output_csv_path, index=False)




# python3 predict.py /Users/tomislavmatanovic/Documents/Melanoma_Lumen/Data/train tic01.csv

#python3 predict.py /Users/tomislavmatanovic/Documents/Melanoma_Lumen/Data/ISIC_2020_Test_Input validation_output.csv
