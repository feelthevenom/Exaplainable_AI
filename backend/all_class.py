import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
#! conda  install --yes seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Torch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height = 256, width = 256, always_apply = True),
        # album.RandomSizedCrop(min_max_height=150,height=256,width = 256,p =1),
        album.OneOf(
            [
                album.HorizontalFlip(p = 1),
                album.VerticalFlip(p = 1),
                album.RandomRotate90(p = 1),
                # album.RandomSizedCrop(min_max_height=150,height=256,width = 256,p =1),
            ],
            p = 0.75,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # album.PadIfNeeded(min_height = 500, min_width = 500, always_apply = True, border_mode = 0),
        album.PadIfNeeded(min_height = 1536, min_width = 1536, always_apply = True, border_mode = 0)
        # album.PadIfNeeded(min_height = 512, min_width = 512, always_apply = True, border_mode = 0)
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """
    Construct preprocessing transform

    Arguments:
        preprocessing_fn (callable): data normalization function (can be specific for each pretrained neural network)
    Returns:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image = preprocessing_fn))
    _transform.append(album.Lambda(image = to_tensor))

    return album.Compose(_transform)





# Useful to shortlist specific classes in datasets with large number of classes
class_names = ['background', 'building']
select_classes=class_names
class_rgb_values= [[0, 0, 0], [255, 255, 255]]
# class_rgb_values= [[255, 255, 255], [0, 0, 0]]

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]



  # Helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize = (20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # Get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize = 20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis = -1)

    return semantic_map

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class BuildingsDataset(torch.utils.data.Dataset):
    """
    Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Arguments:
        images_dir (str) : path to images folder or a single image file
        class_rgb_values (list) : RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose) : data transformation pipeline (e.g., flip, scale, etc.)
        preprocessing (albumentations.Compose) : data preprocessing (e.g., normalization, shape manipulation, etc.)
    """

    def __init__(
            self,
            images_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        if os.path.isdir(images_dir):
            # If it's a directory, get the list of image files
            self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        elif os.path.isfile(images_dir):
            # If it's a file, use it as a single image
            self.image_paths = [images_dir]
        else:
            raise ValueError("Invalid input: images_dir should be a directory or a single image file.")

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read images
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        # Return length
        return len(self.image_paths)

# Center crop padded image / mask to original image dims.
#1500 1500 original
def crop_image(image, target_image_dims=[1536,1536,3]):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]




