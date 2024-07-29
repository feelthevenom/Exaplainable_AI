import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov10', 'yolov10s', pretrained=True)

def detect_cars(image_path):
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)

    # Extract car detections
    car_detections = results.pred[0][results.pred[0][:, -1] == 2]  # Class 2 is 'car' in COCO dataset

    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for det in car_detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return img_rgb, mask

# Load and process your image
image_path = 'car_image.png'  # Replace with your image path
original, car_mask = detect_cars(image_path)

# Create overlay
car_mask_3channel = cv2.cvtColor(car_mask, cv2.COLOR_GRAY2RGB)
overlay = cv2.addWeighted(original, 0.7, car_mask_3channel, 0.3, 0)

# Visualize the results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(car_mask, cmap='gray')
plt.title('Car Detection Mask')
plt.axis('off')

plt.subplot(133)
plt.imshow(overlay)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()

# Save the plot
plt.savefig('car_detection_results.png')

# Display the plot
plt.show()

# Save individual images
cv2.imwrite('overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("Images have been saved.")