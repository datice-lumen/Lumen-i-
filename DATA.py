# ABOUT CODE
# What it does?
# What is input?
# What is expected oputput?
# Do before running:
# What is next?

#project_folder/
#│
#├── train/                         # Folder with raw skin images
#│   ├── image1.jpg
#│   ├── image2.jpg
#│   └── ...
#│
#├── ISIC_2020_Training_GroundTruth.csv   # CSV with labels/metadata
#├── 2020_Challenge_duplicates.csv        # CSV listing duplicate image entries
#├── your_script.py                # This script file

import os
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
from collections import defaultdict
import json
import sys
import time
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count



# INPUT PARAMETERS
PERCENT = 1 # percentage of total available images that you want to preprocess and save (0.0 min, 1.0 max)
final_folder_name = "2all_k_FOLD_dataset_PREPROCESSED2"
TARGET_SIZE = (200, 200) # target size for final images
NUM_FOLDS = 5
PARALLEL = True   # set True if you want to preprocess images in parallel with 70% of cores available (speeds up sugnificantly)




np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imgs_path = os.path.join(BASE_DIR, "train")     # path to folder with all images - raw, given from competition
csv_all_path = os.path.join(BASE_DIR, "ISIC_2020_Training_GroundTruth.csv") # csv containing metadata, given from competition
duplicates_csv_path = os.path.join(BASE_DIR, "2020_Challenge_duplicates.csv") # csv with found duplicates with diff. names
final_folder_path = os.path.join(BASE_DIR, final_folder_name)
stats_path = os.path.join(final_folder_path, "stats_logs.txt")


os.makedirs(os.path.dirname(stats_path), exist_ok=True)


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

def check_duplicate(image_path):
    if image_path in VISITED_FILES or img_name in duplicates_set:
        return True
    else:
        VISITED_FILES.add(image_path)
        return False

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

def control_plot(orig_image, cropped_img, crop_hairless_img):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.title("ORIGINAL IMAGE (1800x1200)")
    #plt.axis("off")
    ##########

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title("CROPPED IMAGE (1200x1200)")
    #plt.axis("off")
    ##########

    plt.subplot(1, 4, 3)
    image_resized1 = cv2.resize(crop_hairless_img, (300, 300), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(cv2.cvtColor(image_resized1, cv2.COLOR_BGR2RGB))
    plt.title("SMALLER NO HAIR (300x300)")
    #plt.axis("off")

    ##########

    plt.subplot(1, 4, 4)
    image_resized2 = cv2.resize(crop_hairless_img, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(cv2.cvtColor(image_resized2, cv2.COLOR_BGR2RGB))
    plt.title("SMALLER NO HAIR (200x200)")
    #plt.axis("off")

    plt.show()
    return


def calculate_ITA_subregions(in_image, verbose, i):
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

    if verbose and i<10:
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






def preprocess(image_path, i, TS = (200, 200), verbose = False):

    #print(f"aprox. {i+1} / 1500 images")

    if os.path.exists(image_path):
        orig_image = cv2.imread(image_path)
    else:
        print(f"Image not found: {image_path}")
        return None

    orig_image = cv2.resize(orig_image, (1200, 800), interpolation=cv2.INTER_AREA) 
    # resizing with 2 steps, to reduce computation in-between steps, and to preserve as much quality in resizing



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

    if verbose and i<10:
        control_plot(orig_image, cropped_img, crop_hairless_img)

    # CALCULATE FITZPATRICK
    #avg_ita, avg_fitz, fitz_dict = calculate_ITA_subregions(small_crop_hairless_img, verbose, i)
    avg_top2_ITA, all8_ita_list = calculate_ITA_subregions(small_crop_hairless_img, verbose, i)


    fn = os.path.basename(image_path)


    data_dict = {
        "image_name": fn,
        "brightest_ITA": avg_top2_ITA,
        "ita_list": all8_ita_list
    }

    return small_crop_hairless_img, data_dict 
    # image and metadatada


# FUNCTION FOR PARALLEL PREPROCESSING
#/////////////////////////////////////

def parallel_preprocess(image_paths, num_processes):
    """ Run image preprocessing in parallel and track progress. """
    
    # Create a Pool of workers to process images in parallel
    pool = multiprocessing.Pool(processes=num_processes)

    # Create the inputs for the preprocessing function (image paths and index)
    inputs = [(img_path, idx) for idx, img_path in enumerate(image_paths)]

    # Map the preprocess function to inputs using pool.map (this runs the function in parallel)
    results = pool.starmap(preprocess, inputs)

    pool.close()
    pool.join()

    return results







# FUNCTIONS FOR FOLDING (k-fold)
#/////////////////////////////////////

#CONTROLLA = defaultdict(int)
def triple_stratified_fold(folds = 5):

    # tsf_df with collumns: "patientID", "real_image_name", "foldID", "target_class"


    data = []
    sub_dict_only1 = {k: v for k, v in patient_dict.items() if v[1] != 0} # extract only patients with "class 1" images  #includes patients with images in BOTH classes
    sub_dict_only0 = {k: v for k, v in patient_dict.items() if k not in sub_dict_only1} # leave only class0 patients

    index_for_class_1 = 0
    sorted_sub_dict_only1 = dict(sorted(sub_dict_only1.items(), key=lambda item: item[1][1], reverse=True))

    for k, v in sorted_sub_dict_only1.items():

        if v[1] > 0: # patients with only CLASS 1 or BOTH

            # add all of his images into final df
            filtered_df = tmp_df[tmp_df["patient_id"] == k]
            for index, row in filtered_df.iterrows():
                #data.append({"patientID": k, "real_image_name": row["real_image_name"], "foldID": index_for_class_1, "target_class": row["label"]})
                data.append({"patientID": k, "image_name": row["image_name"], "foldID": index_for_class_1, "target_class": row["target"]})
                # ####
                # if row["label"] == "1":
                #     #print("JOJO")
                #     CONTROLLA[row["real_image_name"]] = 0
                # ###

            index_for_class_1 = (index_for_class_1 + 1) % folds
        else:
            print("ERROR1!")


    index_0 = 0
    sorted_sub_dict_only0 = dict(sorted(sub_dict_only0.items(), key=lambda item: item[1][0], reverse=True))

    for k, v in sorted_sub_dict_only0.items():
        if v[0] > 0 and v[1] == 0: # patients with just CLASS 0 images

                # add all of his images into final df
                filtered_df = tmp_df[tmp_df["patient_id"] == k]
                for index, row in filtered_df.iterrows():
                    #data.append({"patientID": k, "real_image_name": row["real_image_name"], "foldID": index_0, "target_class": row["label"]})
                    data.append({"patientID": k, "image_name": row["image_name"], "foldID": index_0, "target_class": row["target"]})

                index_0 = (index_0 + 1) % folds
        else:
            print("ERROR0!")

    tsf_df = pd.DataFrame(data)

    return tsf_df




















# MAIN CODE
#/////////////////////////////////////
if __name__ == "__main__":

    log_file = open(stats_path, "w")
    sys.stdout = log_file

    VISITED_FILES = set()

    duplicates_df = pd.read_csv(duplicates_csv_path) 
    duplicates_set = set(duplicates_df['ISIC_id_paired'].tolist())

    df = pd.read_csv(csv_all_path)

    # Choose random rows
    tmp_df = df.sample(n= int(len(df)*PERCENT), random_state=43)
    print("\nNUM SAMPLES BEFORE DUPLICATE CHECK:", len(tmp_df))

    # Check and remove duplicates
    rows_to_drop = []
    for idx, img_name in tmp_df['image_name'].items():  
        if check_duplicate(img_name):
            print(f"Duplicate image: {img_name}, skipping...")
            duplicates_set.remove(img_name)
            rows_to_drop.append(idx)
            # remove that row from tmp_df

    # drop duplicate rows 
    tmp_df = tmp_df.drop(rows_to_drop).reset_index(drop=True)
    tot_samples = len(tmp_df)
    print("\nNUM SAMPLES AFTER DUPLICATE CHECK:", len(tmp_df))

    # Count unique values in the 'target' column
    target_counts = tmp_df['target'].value_counts()

    print()
    print(target_counts)
    print(target_counts[1]/(target_counts[0]+target_counts[1]))


    # some info about patients and their images
    #####

    print("\n\n\n")
    print("--------------------")

    # Initialize dictionary with default values (0,0) for class counts
    patient_dict = defaultdict(lambda: [0, 0])

    # Iterate through the dataframe and count images per class for each patient
    for _, row in tmp_df.iterrows():
        patient_id = row["patient_id"]
        label = int(row["target"])  # Ensure it's an integer (0 or 1)
        patient_dict[patient_id][label] += 1  # Increment class count

    # Convert defaultdict to a normal dictionary (optional)
    patient_dict = dict(patient_dict)

    # Print the first few entries
    # for k, v in list(patient_dict.items())[:15]:
    #     print(f"Patient {k}: Class 0 = {v[0]}, Class 1 = {v[1]}")


    # Patient with max images
    max_patient = max(patient_dict.items(), key=lambda x: sum(x[1]))
    print(f"\nPatient with max images: {max_patient[0]} - Total Images: {sum(max_patient[1])}")


    # Avg images per patient
    total_images = sum(sum(v) for v in patient_dict.values())
    avg_images_per_patient = total_images / len(patient_dict)
    print(f"Average images per patient: {avg_images_per_patient:.2f}")


    # Count and print patients with both class0 and class1 images
    print("--------------------")
    both_classes_patients = 0
    for k, v in patient_dict.items():
        if v[0] > 0 and v[1] > 0:
            print(k, v)
            both_classes_patients += 1
    print(f"Patients with both class 0 and class 1 images: {both_classes_patients}\n\n")



    # REMOVE DOMINANT PATIENTS
    ##########################
    # here we will set threshold of max images per patient, to embrace entropy and prevent overfitting
    # will be removing only class_0 images
    # it also helps balance out classes 0 and 1

    # Filter patients with at least one class 0 image
    filtered_patients0 = {k: v for k, v in patient_dict.items() if v[0] > 0}

    # Sort them by total number of images (class 0), ascending
    sorted_filtered0 = sorted(filtered_patients0.items(), key=lambda x: x[1][0])

    print("BEFORE::")
    print("\tTOTAL SAMPLES ->", len(tmp_df))
    print("\tSAMPLES of class 1 ->", (tmp_df["target"] == 1).sum())

    max_0_im_per_patient = 20
    rows_to_drop = []

    for patient_id, counts in sorted_filtered0:
        # Get all class 0 rows for this patient
        patient_0_rows = tmp_df[(tmp_df['patient_id'] == patient_id) & (tmp_df['target'] == 0)]

        if len(patient_0_rows) > max_0_im_per_patient:

            # Number of rows to remove
            n_to_remove = len(patient_0_rows) - max_0_im_per_patient
            drop_indices = np.random.choice(patient_0_rows.index, size=n_to_remove, replace=False)
            
            rows_to_drop.extend(drop_indices)

    # Drop all at once
    tmp_df = tmp_df.drop(index=rows_to_drop).reset_index(drop=True)

    new_total = len(tmp_df)

    print(f"True count of images in tmp_df after: {len(tmp_df)}")
    print(f"THAT is {(new_total/tot_samples)}")
    tot_samples = len(tmp_df) #update
    print("AFTER::")
    print("\tTOTAL SAMPLES ->", len(tmp_df))
    print("\tSAMPLES of class 1 ->", (tmp_df["target"] == 1).sum())
    print()




##del

    print("\n\n--------------------\n")

    patient_dict = defaultdict(lambda: [0, 0])

    # Iterate through the dataframe and count images per class for each patient
    for _, row in tmp_df.iterrows():
        patient_id = row["patient_id"]
        label = int(row["target"])  # Ensure it's an integer (0 or 1)
        patient_dict[patient_id][label] += 1

    patient_dict = dict(patient_dict)

    # Print the first few entries
    # for k, v in list(patient_dict.items())[:15]:
    #     print(f"Patient {k}: Class 0 = {v[0]}, Class 1 = {v[1]}")


    # Patient with max images
    max_patient = max(patient_dict.items(), key=lambda x: sum(x[1]))
    print(f"\nPatient with max images: {max_patient[0]} - Total Images: {sum(max_patient[1])}")


    # Avg images per patient
    total_images = sum(sum(v) for v in patient_dict.values())
    avg_images_per_patient = total_images / len(patient_dict)
    print(f"Average images per patient: {avg_images_per_patient:.2f}")


    # Count and print patients with both class0 and class1 images
    print("--------------------")
    both_classes_patients = 0
    for k, v in patient_dict.items():
        if v[0] > 0 and v[1] > 0:
            print(k, v)
            both_classes_patients += 1
    print(f"Patients with both class 0 and class 1 images: {both_classes_patients}\n\n")

    # PLOT histogram, y- count of patients, x - num of images per patient
    image_counts = [sum(v) for v in patient_dict.values()]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(image_counts, bins=range(1, max(image_counts) + 2), edgecolor='black', alpha=0.75)

    # Labels and title
    plt.xlabel("Number of Images per Patient")
    plt.ylabel("Count of Patients")
    plt.title("Distribution of Images per Patient")
    xticks = range(1, max(image_counts) + 1)
    plt.xticks(xticks[::2], rotation=90)
    # Show plot
    plt.show()

###


    # FOLDING
    #########

    # from tmp_df, decide folds
    tmp_folded_df = triple_stratified_fold(folds = NUM_FOLDS)

    # Count patients per fold
    patients_per_fold = tmp_folded_df.groupby("foldID")["patientID"].nunique()

    # Sum of images per fold, split by class
    counts_per_fold_class = tmp_folded_df.groupby(["foldID", "target_class"]).size().unstack(fill_value=0)
    counts_per_fold_class["total"] = counts_per_fold_class.sum(axis=1)



    # Print results
    print("---------------------")
    print("\nFinal DataFrame:")
    print(tmp_folded_df.head())

    print("\nPatients per fold:")
    print(patients_per_fold)

    print("\nTotal images per fold (split by class):")
    # Print results
    print(counts_per_fold_class)

    sys.stdout = sys.__stdout__
    log_file.close()

    output_path = os.path.join(BASE_DIR, 'tsf_folds.csv')
    tmp_folded_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    ####
    ###





    # Create the final folder if it doesn't exist
    os.makedirs(final_folder_path, exist_ok=True)

    FINAL_DATA = []

    # Copy images to the new folder
    if PARALLEL == False:
        i=0
        time_100_aprox = -1
        start_time = time.time()

        for index, row in tmp_folded_df.iterrows():
            img_file = row["image_name"] + ".jpg" 
            source_path = os.path.join(BASE_DIR, "train", img_file)

            #print(i+1, "source path:", source_path)
            prep_img, prep_data = preprocess(source_path, i)
            if prep_img is None:
                continue
            
            #prep data append fold_id i class_true
            #patient id!!!
            prep_data["fold_id"] = row["foldID"]
            prep_data["true_class"] = row["target_class"]

            i+=1
            if (i%100==0):
                print(f"\tupdate: {i} / {tot_samples}")
                if (i==100):
                    #save time per 100 examples
                    time_100_aprox = time.time() - start_time # will be used for aprox.
                else:
                    #calculate time remaining aprox
                    time_remaining = ((tot_samples - i) / 100) * time_100_aprox
                    print(f"\taprox time remaining: {int(time_remaining // 60)} min {int(time_remaining % 60)} sec\n")
                


            #check if destination_folder exists, if not, create it
            destination_folder = os.path.join(final_folder_path, f"{row["foldID"]}_fold")
            os.makedirs(destination_folder, exist_ok=True)
            cv2.imwrite(os.path.join(destination_folder, f"pre_{img_file}"), prep_img)
            FINAL_DATA.append(prep_data)
    
    else: # PARALLELIZED

        # IMAGE PREPROCESSING - PARALLELIZED
        images_to_process = [ os.path.join(BASE_DIR, "train", name + ".jpg") for name in tmp_folded_df['image_name']]
        print("images_to_process:", len(images_to_process))
        #num_processes = multiprocessing.cpu_count()  # Use all available cores
        num_processes = int(np.ceil(multiprocessing.cpu_count() * 0.7)) # Use around 70% of core available

        # Preprocess images in parallel
        results = parallel_preprocess(images_to_process, num_processes)
        print("results:", len(results))

        for tup in results:
            prep_img, metadata = tup

            # Find the matching row in tmp_folded_df
            match_row = tmp_folded_df[tmp_folded_df['image_name'] == metadata["image_name"].removesuffix(".jpg")]

            metadata["patient_id"] = match_row.iloc[0]["patientID"] #i need you to find row in tmp_folded_df that matches metadata["image_name"] without .jpg and from that row extract collumn "patientID"
            metadata["fold_id"] = match_row.iloc[0]["foldID"]
            metadata["true_class"] = match_row.iloc[0]["target_class"]

        
            #check if destination_folder exists, if not, create it
            destination_folder = os.path.join(final_folder_path, f"{match_row.iloc[0]["foldID"]}_fold")
            os.makedirs(destination_folder, exist_ok=True)
            cv2.imwrite(os.path.join(destination_folder, f"pre_{metadata["image_name"]}"), prep_img)
            FINAL_DATA.append(metadata)



    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(elem) for elem in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

        
    # SAVE metadata as json
    json_path = os.path.join(final_folder_path, "metadata_all_fold.json")
    with open(json_path, "w") as json_file:
        json.dump(convert_numpy(FINAL_DATA), json_file, indent=4)


    print("\nDONE:")
    print("#############################")
    print("Images successfully preprocessed to:", final_folder_path)
    print("Metadata successfully copied to:", json_path)
    print("Stats/Logs successfully copied to:", stats_path)
    if PARALLEL:
        print(f"-- USED {num_processes} CORES OUT OF {multiprocessing.cpu_count()} AVAILABLE --")


    # Whats next, take those folds, upload them to google drive, so that you can run training for cnn on those preprocessed images
    # Alternatevly, run localy as notebook

