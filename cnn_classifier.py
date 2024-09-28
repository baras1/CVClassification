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
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN Classifier
class CNNClassifier:
    def __init__(self, num_classes, lr=0.001, batch_size=32, num_epochs=10):
        self.model = CNNModel(num_classes=num_classes).cuda()
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
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    def predict(self, X):
        self.model.eval()
        X = X.cuda()
        with torch.no_grad():
            outputs = self.model(X)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X = X.cuda()
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
