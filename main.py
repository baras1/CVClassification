import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Function to load images and convert to PyTorch tensors
def load_images_from_csv(csv_file, image_folder, has_labels=True):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # If the CSV contains labels, separate file paths and labels
    if has_labels:
        filepaths = df.iloc[:, 0]  # Assuming the first column contains file paths
        labels = df.iloc[:, 1]     # Assuming the second column contains labels
    else:
        filepaths = df.iloc[:, 0]  # For test dataset, only file paths are present
    
    # To store the images and labels (if available)
    images = []
    labels_list = []

    # Transform images to torch tensor
    transform = transforms.ToTensor()  # This will transform to PyTorch tensors

    # Load each image
    for filepath in filepaths:
        # Create full path to the image
        image_path = os.path.join(image_folder, filepath)
        print("working")
        
        # Open image using PIL
        img = Image.open(image_path).convert('RGB')  # Convert to RGB format
        
        # Convert to a torch tensor (or you can convert to a numpy array)
        img_tensor = transform(img)  # This will create a (C, H, W) tensor

        # Append the image to the list
        images.append(img_tensor)

        # If training dataset, append the corresponding label
        if has_labels:
            labels_list.append(labels[filepaths == filepath].values[0])  # Get the label

    if has_labels:
        return torch.stack(images), torch.tensor(labels_list)  # Return both images and labels as tensors
    else:
        return torch.stack(images)  # For test dataset, just return images

# Usage example:
# For training data
train_images, train_labels = load_images_from_csv('Training_set.csv', 'train', has_labels=True)

# For test data (without labels)
test_images = load_images_from_csv('Testing_set.csv', 'test', has_labels=False)

