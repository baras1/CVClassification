import torch
import numpy as np
from cnn_classifier import CNNClassifier
from cnn_np_classifier import CNNClassifierNumpy
from mlp_classifier import MLPClassifier
import os

class ClassifierManager:
    def __init__(self, model_type, save_model=True, load_model=False, **kwargs):
        self.model_type = model_type
        self.save_model = save_model
        self.load_model_flag = load_model

        if model_type == 'cnn_numpy':
            self.classifier = CNNClassifierNumpy(**kwargs)
        elif model_type == 'cnn':
            self.classifier = CNNClassifier(**kwargs)
        elif model_type == 'mlp':
            self.classifier = MLPClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y, model_path=None):
        # Optionally load model if load_model_flag is set and model_path is provided
        if self.load_model_flag and model_path:
            print(f"Loading model from {model_path}")
            self.load_model(model_path)
        else:
            print("Training model from scratch...")
            self.classifier.fit(X, y)

        # Optionally save the model if save_model is set and model_path is provided
        if self.save_model and model_path:
            print(f"Saving model to {model_path}")
            self.save_model_method(model_path)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_with_visualization(self, X, y=None):
        return self.classifier.predict_with_visualization(X, y)

    #saving models
    def save_model_method(self, model_path):
        # Ensure the directory exists
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.model_type == 'cnn':
            torch.save(self.classifier.model.state_dict(), f"{model_path}.pth")
        elif self.model_type in ['mlp', 'cnn_numpy']:
            np.save(f'{model_path}_weights1.npy', self.classifier.model.weights1)
            np.save(f'{model_path}_bias1.npy', self.classifier.model.bias1)
            np.save(f'{model_path}_weights2.npy', self.classifier.model.weights2)
            np.save(f'{model_path}_bias2.npy', self.classifier.model.bias2)
            np.save(f'{model_path}_weights3.npy', self.classifier.model.weights3)
            np.save(f'{model_path}_bias3.npy', self.classifier.model.bias3)

    # Loading models
    def load_model(self, model_path):
        if self.model_type == 'cnn':
            # Load the saved state_dict
            state_dict = torch.load(model_path)
            
            # Get the current model's state_dict
            model_state_dict = self.classifier.model.state_dict()

            # Filter out any keys in the loaded state_dict that don't exist in the current model
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            # Update the model's state_dict
            model_state_dict.update(filtered_state_dict)
            
            # Load the filtered state_dict into the model
            self.classifier.model.load_state_dict(model_state_dict)

        elif self.model_type in ['mlp', 'cnn_numpy']:
            self.classifier.model.weights1 = np.load(f'{model_path}_weights1.npy')
            self.classifier.model.bias1 = np.load(f'{model_path}_bias1.npy')
            self.classifier.model.weights2 = np.load(f'{model_path}_weights2.npy')
            self.classifier.model.bias2 = np.load(f'{model_path}_bias2.npy')
            self.classifier.model.weights3 = np.load(f'{model_path}_weights3.npy')
            self.classifier.model.bias3 = np.load(f'{model_path}_bias3.npy')
