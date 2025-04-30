# Lumen Data Science - Fair Melanoma Detection Challenge

This repository contains our solution to the Fair Melanoma Detection Challenge using dermatoscopic images.

## Goal

Develop a reliable and fair machine learning model that classifies skin lesions as **benign or malignant**, focusing on performance **across skin tones**.

## Contents

1. Environment Setup & Configuration  
   - [1.1 Installing Dependencies](#11-installing-dependencies)  
   - [1.2 Using `default.yaml` Configuration File](#12-using-defaultyaml-configuration-file)  
   - [1.3 Dataset Placement and Folder Structure](#13-dataset-placement-and-folder-structure)  

2. Training Pipeline  
   - [2.1 Launching Training from Script](#21-launching-training-from-script)  
   - [2.2 Saving Checkpoints and Logs](#22-saving-checkpoints-and-logs)  

3. Evaluation & Fairness  
   - [3.1 Validation Set Evaluation](#31-validation-set-evaluation)  
   - [3.2 Fairness Across Skin Tones](#32-fairness-across-skin-tones)  
   - [3.3 Visualization and Output Files](#33-visualization-and-output-files)  

4. Inference  
   - [4.1 Predicting on New Images](#41-predicting-on-new-images)  
   - [4.2 Output Format (`validation_output.csv`)](#42-output-format-validation_outputcsv)  

5. Reproducibility & Submission  
   - [5.1 Running from Pretrained Model](#51-running-from-pretrained-model)  
   - [5.2 Submission Script and Output Format](#52-submission-script-and-output-format)

6. Additional Documentation
   - 6.1 [Project Report](reports/project_documentation.pdf)  
   - 6.2 [Technical Documentation](reports/technical_documentation.pdf)


## 1. Environment Setup & Configuration

### 1.1 Installing Dependencies

This project relies on TensorFlow, PyTorch, and several libraries for data handling, visualization, and image processing.

To install all necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### 1.2 Using `default.yaml` Configuration File

The `default.yaml` file contains all configurable settings used for training, validation, and inference.  
Instead of hardcoding parameters in scripts, this file provides a centralized way to control:

- Dataset paths (train/validation folders, metadata file)
- Model architecture and parameters
- Training hyperparameters (epochs, batch size, learning rate, etc.)
- Callbacks and optimization settings
- Output checkpoint naming and save directory
- Inference threshold

Example usage (loaded inside a script):

```python
import yaml

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
```

### 1.3 Dataset Placement and Folder Structure

The project expects the dataset to be organized into separate `train/` and `val/` folders inside a `data/` directory.  
Each folder should contain image files, and a shared metadata file should be placed in the same directory.

Recommended folder structure:


The training dataset used in this project can be downloaded from the [ISIC 2020 Challenge website](https://challenge2020.isic-archive.com/).
The `metadata_all_fold.json` file was generated through a custom preprocessing step and includes relevant information such as image names and skin tone metadata.

The paths to these folders and the metadata file are defined inside `configs/default.yaml`. 


## 2. Training Pipeline

### 2.1 Launching Training from Script

### 2.2 Saving Checkpoints and Logs 
