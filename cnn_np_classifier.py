import numpy as np
import random
import matplotlib.pyplot as plt
from cnn_np_model import CNN

class CNNClassifierNumpy:
    def __init__(self, input_shape, num_classes, learning_rate=0.001, epochs=1000):
        self.model = CNN(input_shape, num_classes)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.model.train(X, y, epochs=self.epochs, learning_rate=self.learning_rate)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_visualization(self, X, y=None):
        predictions = self.model.predict(X)

        # Select 10 random examples for visualization
        indices = random.sample(range(X.shape[0]), 10)
        if y is not None:
            y = y[indices]
        X_images = X[indices]

        # Reshape to original image format (e.g., 128x128x3)
        X_images_reshaped = X_images.reshape(-1, 3, 128, 128)

        # Plot images with predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            image = X_images_reshaped[i].transpose(1, 2, 0)  # (H, W, C)
            axes[i].imshow(image)
            if y is not None:
                axes[i].set_title(f"Pred: {predictions[idx]}, Actual: {y[i]}")
            else:
                axes[i].set_title(f"Pred: {predictions[idx]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return predictions
