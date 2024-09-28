from cnn_np_classifier import CNNClassifierNumpy
from cnn_classifier import CNNClassifier
from mlp_classifier import MLPClassifier

class ClassifierManager:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        if model_type == 'cnn_numpy':
            self.classifier = CNNClassifierNumpy(**kwargs)
        elif model_type == 'cnn':
            self.classifier = CNNClassifier(**kwargs)
        elif model_type == 'mlp':
            self.classifier = MLPClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_with_visualization(self, X, y=None):
        return self.classifier.predict_with_visualization(X, y)
