import numpy as np

# Helper functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    p = softmax(predictions)
    log_likelihood = -np.log(p[range(m), labels])
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_derivative(predictions, labels):
    m = labels.shape[0]
    grad = softmax(predictions)
    grad[range(m), labels] -= 1
    grad = grad / m
    return grad

# Convolution layer
def conv2d(X, filters, stride=1, padding=1):
    # X: input batch of images, filters: convolution filters
    batch_size, n_c, h_prev, w_prev = X.shape
    n_filters, d_filter, f, _ = filters.shape

    h = (h_prev - f + 2 * padding) // stride + 1
    w = (w_prev - f + 2 * padding) // stride + 1
    Z = np.zeros((batch_size, n_filters, h, w))

    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            X_slice = X_padded[:, :, i:i+f, j:j+f]
            for k in range(n_filters):
                Z[:, k, i//stride, j//stride] = np.sum(X_slice * filters[k], axis=(1, 2, 3))  # Convolution

    return Z

# Max pooling layer
def maxpool2d(X, f=2, stride=2):
    batch_size, n_c, h_prev, w_prev = X.shape
    h = (h_prev - f) // stride + 1
    w = (w_prev - f) // stride + 1
    Z = np.zeros((batch_size, n_c, h, w))

    for i in range(0, h_prev, stride):
        for j in range(0, w_prev, stride):
            Z[:, :, i//stride, j//stride] = np.max(X[:, :, i:i+f, j:j+f], axis=(2, 3))

    return Z

# Flatten layer
def flatten(X):
    return X.reshape(X.shape[0], -1)

class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Initialize filters for convolution layers
        self.filters1 = np.random.randn(8, input_shape[0], 3, 3) / 9  # 8 filters, kernel 3x3
        self.filters2 = np.random.randn(16, 8, 3, 3) / 9  # 16 filters

        # Fully connected layer weights
        fc_input_dim = 16 * 32 * 32  # Adjust based on image size and architecture
        self.fc1_weights = np.random.randn(fc_input_dim, 128) / np.sqrt(fc_input_dim)
        self.fc2_weights = np.random.randn(128, num_classes) / np.sqrt(128)

    def forward(self, X):
        # Forward pass: Convolution -> ReLU -> Max Pool -> Convolution -> ReLU -> Max Pool -> Fully connected layers

        # First conv layer
        self.z1 = conv2d(X, self.filters1, stride=1, padding=1)
        self.a1 = relu(self.z1)
        self.pool1 = maxpool2d(self.a1, f=2, stride=2)

        # Second conv layer
        self.z2 = conv2d(self.pool1, self.filters2, stride=1, padding=1)
        self.a2 = relu(self.z2)
        self.pool2 = maxpool2d(self.a2, f=2, stride=2)

        # Flatten and fully connected layers
        self.flat = flatten(self.pool2)
        self.fc1 = relu(np.dot(self.flat, self.fc1_weights))
        self.fc2 = np.dot(self.fc1, self.fc2_weights)

        return self.fc2

    def backward(self, X, y, learning_rate=0.001):
        # Compute gradients and update filters/weights
        d_loss_output = cross_entropy_derivative(self.fc2, y)

        # Backpropagate through fully connected layers
        d_fc2 = d_loss_output
        d_fc1 = np.dot(d_fc2, self.fc2_weights.T) * relu_derivative(self.fc1)

        d_fc2_weights = np.dot(self.fc1.T, d_fc2)
        d_fc1_weights = np.dot(self.flat.T, d_fc1)

        self.fc2_weights -= learning_rate * d_fc2_weights
        self.fc1_weights -= learning_rate * d_fc1_weights

        # Backprop through pooling and conv layers (not fully implemented here due to complexity)
        # To extend, you need to backpropagate through the conv layers as well
        pass

    def train(self, X, y, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = cross_entropy_loss(predictions, y)

            # Backward pass
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(softmax(logits), axis=1)
