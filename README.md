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
   - 2.1 Launching Training from Script  
   - 2.2 Saving Checkpoints and Logs  

3. Evaluation & Fairness
   - 3.1 Validation Set Evaluation  
   - 3.2 Fairness Across Skin Tones  
   - 3.3 Visualization and Output Files  

4. Inference
   - 4.1 Predicting on New Images  
   - 4.2 Output Format (`validation_output.csv`)  

5. Reproducibility & Submission
   - 5.1 Running from Pretrained Model  
   - 5.2 Submission Script and Output Format  

6. Repository Structure

7. Additional Documentation
   - 7.1 [Project Report](reports/project_documentation.pdf)  
   - 7.2 [Technical Documentation](reports/technical_documentation.pdf)

## 1. Environment Setup & Configuration

### 1.1 Installing Dependencies

This project relies on TensorFlow, PyTorch, and several libraries for data handling, visualization, and image processing.

To install all necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## 1.2 Using `default.yaml` Configuration File

## 1.3 Dataset Placement and Folder Structure

