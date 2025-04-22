# import (maybe) correct version of torch
import os # tebi ovo mozda nece trebat
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch #2.6.0+cu124 bi trebao (ja imam 2.6.0 i dela sve)
import torch.nn as nn
import torchvision.models as models
##from tensorflow.keras.preprocessing.image import load_img, img_to_array
print(torch.__version__) #(ja imam 2.6.0 i dela sve)
# ak jos nemas torch, onda skini tu verziju najbolje




# ovo je samo copy paste funkcija za preprocessing (nis zanimljivo)
# FUNCTIONS FOR PREPROCESSING
#/////////////////////////////////////


def check_resize(in_image):
    # check conditions for further preprocessing
    height, width = in_image.shape[:2]  # Get dimensions
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
        #avg_ita += ITA
        all8_ita_list.append(ITA)

    avg_top2_ITA = sum(sorted(all8_ita_list, reverse=True)[:2]) / 2


    return avg_top2_ITA, all8_ita_list 

def get_fitzpatrick(ita):
        if ita > 55: return 1 #"I (Very Light)"
        elif ita > 41: return 2 #"II (Light)"
        elif ita > 28: return 3 #"III (Intermediate)"
        elif ita > 19: return 4 #"IV (Tan)"
        elif ita > 10: return 5 #"V (Brown)"
        else: return 6 #"VI (Dark Brown/Black)"




def preprocess(orig_image, TS = (224, 224), verbose = False):

    # MSM DA NE TREBA
    # if (check_resize(orig_image) == False ):
    #     print("Input image isnt either low quality or is not in 1.5 ratio!")
    #     return None

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


    # CALCULATE FITZPATRICK
    #avg_ita, avg_fitz, fitz_dict = calculate_ITA_subregions(small_crop_hairless_img, verbose, i)
    avg_top2_ITA, all8_ita_list = calculate_ITA_subregions(small_crop_hairless_img, verbose)




    data_dict = {
        "brightest_ITA": avg_top2_ITA,
        "ita_list": all8_ita_list
    }

    return small_crop_hairless_img, data_dict 

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










# OKE, ovdje je dio s legit stvarima, s NEURONSKOM
# moras definirat isti model kakav ja koristim (to mu je kostur, ali fale mu tezine koje su nadene treningom)
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomEfficientNet, self).__init__()
        # Load the EfficientNet-B0 model (pre-trained)
        self.model = models.efficientnet_b0(pretrained=True)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            # OVAJ DIO KODA (ARHITEKTURA CE SE VJV JOS MIJENJATI)
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),  
            nn.SiLU(), #nn.ReLU(),
            nn.Dropout(0.25), 
            nn.Linear(256, 1) # raw logits
        )

    def forward(self, x):
        return self.model(x)


model = CustomEfficientNet() 
model.load_state_dict(torch.load('/Users/tomislavmatanovic/Documents/Melanoma_Lumen/live_inter/dummy_model_state_dict.pth', map_location=torch.device('cpu'))) #moras ovo nastimat
# sada smo u prazni model ucitali trenirane tezine, to je to
model.eval() 



# ucitavanje i pretprocesiranje ulazne slike
# nez kak ces ovo implementirat, vjv neces preko patha ucitavat, al ugl trebalo bi biti u cv2 obliku
image_path = "/Users/tomislavmatanovic/Documents/Melanoma_Lumen/Data/train/ISIC_0082934.jpg"
if os.path.exists(image_path):
    orig_image = cv2.imread(image_path)
else:
    print(f"\nImage not found: {image_path}n")

preprocessed_img, metadtada_dict = preprocess(orig_image, TS = (224, 224), verbose = False)
# metadata ak ce ti trebati za nesto mozda (aproxiran fitzpatrick)
input_tensor = prepare_tensor_for_model(preprocessed_img)


# ovo je predikcija
with torch.no_grad():
    print("---------------------")
    output = model(input_tensor)
    print("Raw logit:", output)
    sigm_prediction = torch.sigmoid(output).item() 
    print("After sigmoid:", sigm_prediction)
    pred_class = int(sigm_prediction >= 0.5)
    print("Predicted class:", pred_class)
    print("ALL GOOD BOSS! ;)")


# meni na kompu bude:
# Raw logit: tensor([[1.4311]])
# After sigmoid: 0.8070749640464783
# Predicted class: 1
# ALL GOOD BOSS! ;)
