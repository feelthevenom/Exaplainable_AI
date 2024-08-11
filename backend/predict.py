import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from all_class import *
import warnings
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

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
model = torch.load(model_path, map_location=DEVICE)
model.to(DEVICE)
model.eval()

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

# Updated colour_code_segmentation function
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = np.zeros((*image.shape, 3), dtype=np.uint8)
    for i, colour in enumerate(colour_codes):
        x[image == i] = colour
    return x

for idx in range(len(test_dataset)):
    try:
        image = test_dataset[idx]
        image_vis = test_dataset_vis[idx].astype('uint8')
        
        print(f"Original image shape: {image.shape}")
        print(f"Visualization image shape: {image_vis.shape}")
        
        if image is None or image_vis is None:
            print(f"Failed to load image at index {idx}. Skipping...")
            continue
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        
        print(f"Input tensor shape: {x_tensor.shape}")

        # Predict test image
        with torch.no_grad():
            pred_mask = model(x_tensor)
        
        print(f"Pred mask shape: {pred_mask.shape}")
        print(f"Pred mask min: {pred_mask.min()}, max: {pred_mask.max()}")
        
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        
        print(f"Pred mask shape after squeeze: {pred_mask.shape}")
        print(f"Pred mask values: min={pred_mask.min()}, max={pred_mask.max()}")

        # Apply softmax to get probabilities
        pred_mask = np.exp(pred_mask) / np.sum(np.exp(pred_mask), axis=0)
        
        print(f"Pred mask after softmax: min={pred_mask.min()}, max={pred_mask.max()}")

        # Get the class with highest probability for each pixel
        pred_mask = np.argmax(pred_mask, axis=0)

        print(f"Final pred mask shape: {pred_mask.shape}")
        print(f"Final pred mask values: min={pred_mask.min()}, max={pred_mask.max()}")

        # Get prediction channel corresponding to affected class
        pred_affected_heatmap = (pred_mask == select_classes.index('affected')).astype(np.float32)

        # Convert pred_mask to RGB
        pred_mask_rgb = colour_code_segmentation(pred_mask, select_class_rgb_values)

        # Ensure image_vis and pred_mask_rgb have the same shape
        image_vis = cv2.resize(image_vis, (256, 256))
        pred_mask_rgb = cv2.resize(pred_mask_rgb, (256, 256))

        # Stack images horizontally
        stacked_image = np.hstack([image_vis, pred_mask_rgb])

        # Save the stacked image
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR))

        # Visualize using matplotlib
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(image_vis)
        plt.axis('off')
        
        plt.subplot(132)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask_rgb)
        plt.axis('off')
        
        plt.subplot(133)
        plt.title('Affected Heatmap')
        plt.imshow(pred_affected_heatmap, cmap='hot', interpolation='nearest')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_preds_folder, f"visualization_{idx}.png"))
        plt.close()

    except Exception as e:
        print(f"An error occurred processing image {idx}: {str(e)}")
        import traceback
        traceback.print_exc()

print("Prediction completed. Check the 'sample_predictions' folder for results.")