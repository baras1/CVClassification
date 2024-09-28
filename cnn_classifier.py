import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Define fully connected layers (without input size, we'll compute it dynamically)
        self.fc1 = None  # We'll initialize this later once we know the size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # If the fully connected layer isn't initialized, compute and set it
        if self.fc1 is None:
            self._set_fc1_layer(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _set_fc1_layer(self, x):
        """
        Dynamically initialize the first fully connected layer based on the input size.
        This method sets self.fc1.
        """
        num_features = x.size(1) * x.size(2) * x.size(3)  # Compute the flattened size
        self.fc1 = nn.Linear(num_features, 128).to(x.device)  # Now we can initialize fc1


# CNN Classifier
class CNNClassifier:
    def __init__(self, num_classes, lr=0.001, batch_size=32, num_epochs=7):
        # Check if a GPU is available and use it, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CNNModel(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y):
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def predict_with_visualization(self, X, y=None, num_samples=10):
        """
        Predict and visualize a few samples from the dataset.
        X: Input tensor of images
        y: Actual labels (optional)
        num_samples: Number of samples to visualize
        """
        self.model.eval()
        X = X.to(self.device)

        # Select random samples to visualize
        indices = random.sample(range(X.size(0)), num_samples)
        selected_images = X[indices]

        with torch.no_grad():
            outputs = self.model(selected_images)
            _, preds = torch.max(outputs, 1)

        preds = preds.cpu().numpy()
        selected_images = selected_images.cpu()

        if y is not None:
            actual_labels = y[indices].cpu().numpy()

        # Plot the images with their predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Create a 2x5 grid of subplots
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            image = selected_images[i].permute(1, 2, 0).numpy()  # Change shape from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            if y is not None:
                axes[i].set_title(f"Pred: {preds[i]}, Actual: {actual_labels[i]}")
            else:
                axes[i].set_title(f"Pred: {preds[i]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
