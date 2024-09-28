import numpy as np
import random
import matplotlib.pyplot as plt
from mlp_model import MLP

class MLPClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, epochs=1000):
        self.model = MLP(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.model.train(X, y, epochs=self.epochs, learning_rate=self.learning_rate)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_visualization(self, X, y=None):
        """
        Predict the class labels for the input dataset and plot 10 random examples.
        Args:
            X (numpy.ndarray): Input data (flattened).
            y (numpy.ndarray): Target labels (Optional, used for comparing).
        Returns:
            numpy.ndarray: Predicted class labels.
        """
        predictions = self.model.predict(X)
        
        # Select 10 random examples
        indices = random.sample(range(X.shape[0]), 10)
        if y is not None:
            y = y[indices]
        X_images = X[indices]

        # Reshape X back to images for visualization (assuming they were 3-channel, 128x128 before flattening)
        X_images_reshaped = X_images.reshape(-1, 3, 128, 128)  # Adjust to the correct shape

        # Plot images with predicted and actual labels
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            image = X_images_reshaped[i].transpose(1, 2, 0)  # Convert to (H, W, C)
            axes[i].imshow(image)
            if y is not None:
                axes[i].set_title(f"Pred: {predictions[idx]}, Actual: {y[i]}")
            else:
                axes[i].set_title(f"Pred: {predictions[idx]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return predictions
