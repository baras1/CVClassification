import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from classifier_manager import ClassifierManager
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Function to upload the model to Google Drive
def upload_model_to_google_drive(model_path):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Authenticates the user
    drive = GoogleDrive(gauth)
    
    # Create Google Drive file and upload the model
    model_file = drive.CreateFile({'title': os.path.basename(model_path)})
    model_file.SetContentFile(model_path)
    model_file.Upload()
    print(f"Model uploaded to Google Drive: {model_file['id']}")

# Function to load images and convert to PyTorch tensors for CNN or NumPy arrays for MLP
def load_images_from_csv(csv_file, image_folder, has_labels=True, use_numpy=False):
    df = pd.read_csv(csv_file)
    label_to_index = {}

    if has_labels:
        filepaths = df.iloc[:, 0]  # Assuming the first column contains file paths
        labels = df.iloc[:, 1]     # Assuming the second column contains labels
        label_to_index = {label: idx for idx, label in enumerate(sorted(df.iloc[:, 1].unique()))}
        labels = labels.map(label_to_index)  # Convert string labels to integer indices
    else:
        filepaths = df.iloc[:, 0]  # For test dataset, only file paths are present

    images = []
    labels_list = []

    # Choose between PyTorch tensors or NumPy arrays
    transform = transforms.ToTensor() if not use_numpy else None

    for filepath in filepaths:
        image_path = os.path.join(image_folder, filepath)
        img = Image.open(image_path).convert('RGB')  # Convert to RGB format
        
        if use_numpy:
            img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
            img_np = img_np.transpose(2, 0, 1)  # Change to (C, H, W)
            images.append(img_np)
        else:
            img_tensor = transform(img)  # Transform to PyTorch tensor
            images.append(img_tensor)

        if has_labels:
            labels_list.append(labels[filepaths == filepath].values[0])

    if use_numpy:
        images = np.stack(images)
        labels = np.array(labels_list)
    else:
        images = torch.stack(images)
        labels = torch.tensor(labels_list, dtype=torch.long)

    return images, labels, len(label_to_index)


# Function to load the model either locally or from Google Drive
def load_model_from_drive_or_local(model_path, from_drive=False):
    if from_drive:
        # Initialize Google Drive authorization
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        
        # Assuming model is uploaded in drive and get model file
        model_file = drive.CreateFile({'id': 'YOUR_GOOGLE_DRIVE_FILE_ID'})  # Replace with actual ID
        model_file.GetContentFile(model_path)  # Download the file
        print(f"Loaded model from Google Drive: {model_path}")
    else:
        if os.path.exists(model_path):
            print(f"Loaded model locally from: {model_path}")
        else:
            raise FileNotFoundError(f"The model {model_path} does not exist locally.")
    return model_path


def main():
    print("Running the classification model...")
    model_type = 'cnn'  # Options: 'cnn', 'mlp', 'cnn_numpy'

    # Choose whether to save and/or load models
    save_model = True  # Set to True if you want to save the model after training
    load_model = False  # Set to True if you want to load an existing model
    load_from_drive = False  # Set True if you want to load from Google Drive

    # Load data
    use_numpy = model_type != 'cnn'
    train_images, train_labels, num_classes = load_images_from_csv('butterflies/Training_set.csv', 'butterflies/train', has_labels=True, use_numpy=use_numpy)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, random_state=42, test_size=0.25)
    print("Loaded the images")

    model_path = f'models/{model_type}_model'

    if load_model:
        # Load the model either from Google Drive or local storage
        model_path = load_model_from_drive_or_local(model_path, from_drive=load_from_drive)

    if model_type == 'cnn_numpy':
        input_shape = train_images.shape[1:]  # Shape of a single image
        classifier = ClassifierManager(model_type='cnn_numpy', save_model=save_model, 
                                       load_model=load_model, input_shape=input_shape, 
                                       num_classes=num_classes, epochs=1000)
        classifier.fit(train_images, train_labels, model_path)
        classifier.classifier.predict_with_visualization(val_images, val_labels)

    elif model_type == 'mlp':
        input_size = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]  # Flatten for MLP
        train_images_flat = train_images.reshape(train_images.shape[0], -1)
        val_images_flat = val_images.reshape(val_images.shape[0], -1)
        classifier = ClassifierManager(model_type='mlp', save_model=save_model, 
                                       load_model=load_model, input_size=input_size, 
                                       hidden_size=128, output_size=num_classes)
        print("Begin Fitting")
        classifier.fit(train_images_flat, train_labels, model_path)
        print("Finished Fitting")
        classifier.classifier.predict_with_visualization(val_images_flat, val_labels)

    elif model_type == 'cnn':
        classifier = ClassifierManager(model_type='cnn', save_model=save_model, load_model=load_model, num_classes=num_classes)
        print("Begin Fitting")
        classifier.fit(train_images, train_labels, model_path)
        print("Finished Fitting")
        pred_results = classifier.predict(val_images)
        classifier.predict_with_visualization(val_images, val_labels)

    print(classification_report(val_labels, pred_results))

    # Upload to Google Drive if the model was saved locally
    if save_model:
        upload_model_to_google_drive(model_path)


if __name__ == "__main__":
    main()
