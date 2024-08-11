import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from all_class import *
import warnings
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# New imports for GradCAM
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Usage
model_path = 'Plant_disease_UNet.pth'
test_images_dir = 'images_dir/test_images'
output_path = 'prediction_output.png'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it's in the correct directory.")

# Load the entire model
warnings.warn("Loading the model with full pickle support. Ensure you trust the source of the model file.", UserWarning)
# Load the model
model = torch.load(model_path, map_location=DEVICE)
model.to(DEVICE)
model.eval()


# Print model layers
print("Model layers:")
for name, module in model.named_modules():
    print(f"- {name}")

# Ask user for target layer
target_layer_name = input("Enter the name of the target layer for GradCAM: ")
target_layer = dict([*model.named_modules()])[target_layer_name]
grad_cam = SegmentationGradCAM(model, target_layer)

# Check if test_images directory exists
if not os.path.exists(test_images_dir):
    raise FileNotFoundError(f"Directory '{test_images_dir}' not found. Please ensure the path is correct.")

# Get list of image files
test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not test_images:
    raise ValueError(f"No image files found in '{test_images_dir}'. Please ensure there are images in this directory.")

test_dataset = PlantDiseaseDataset(
    test_images,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)

# test dataset for visualization (without preprocessing transformations)
test_dataset_vis = PlantDiseaseDataset(
    test_images,
    augmentation=get_validation_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

sample_preds_folder = 'sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)


for idx in range(len(test_dataset)):
    try:
        image = test_dataset[idx]
        image_vis = test_dataset_vis[idx].astype('uint8')
        
        if image is None or image_vis is None:
            print(f"Failed to load image at index {idx}. Skipping...")
            continue
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        
        # Predict test image
        with torch.no_grad():
            pred_mask = model(x_tensor)
        
        print(f"Prediction shape: {pred_mask.shape}")
        
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        
        # Apply softmax to get probabilities
        pred_mask = np.exp(pred_mask) / np.sum(np.exp(pred_mask), axis=0)
        
        # Get the class with highest probability for each pixel
        pred_mask = np.argmax(pred_mask, axis=0)

        # Get prediction channel corresponding to affected class
        pred_affected_heatmap = (pred_mask == select_classes.index('affected')).astype(np.float32)

        # Convert pred_mask to RGB
        pred_mask_rgb = colour_code_segmentation(pred_mask, select_class_rgb_values)

        # Ensure image_vis and pred_mask_rgb have the same shape
        image_vis = cv2.resize(image_vis, (256, 256))
        pred_mask_rgb = cv2.resize(pred_mask_rgb, (256, 256))

        # Generate GradCAM heatmap
        heatmap = grad_cam.generate_heatmap(x_tensor, target_class=1)  # 1 for 'affected' class
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(image_vis, 0.6, heatmap, 0.4, 0)

        # Stack images horizontally
        stacked_image = np.hstack([image_vis, pred_mask_rgb, superimposed_img])

        # Save the stacked image
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_gradcam_{idx}.png"), cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR))


    except Exception as e:
        print(f"An error occurred processing image {idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        