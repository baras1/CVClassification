import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function for output layer
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix: subtract max
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    p = softmax(predictions)
    log_likelihood = -np.log(p[range(m), labels])
    loss = np.sum(log_likelihood) / m
    return loss

# Derivative of cross-entropy loss with softmax
def cross_entropy_derivative(predictions, labels):
    m = labels.shape[0]
    grad = softmax(predictions)
    grad[range(m), labels] -= 1
    grad = grad / m
    return grad

class MLP():
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias2 = np.zeros((1, hidden_size))
        self.weights3 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias3 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.output = self.z3  # logits (before softmax)
        
        return self.output

    def backward(self, X, y, learning_rate=0.001):
        # Backward pass and weight update

        # Compute the derivative of loss w.r.t the output (cross-entropy + softmax)
        d_loss_output = cross_entropy_derivative(self.output, y)

        # Backpropagation through second hidden layer
        d_z3 = d_loss_output
        d_w3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_a2 = np.dot(d_z3, self.weights3.T)
        d_z2 = d_a2 * relu_derivative(self.z2)
        d_w2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_w1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights3 -= learning_rate * d_w3
        self.bias3 -= learning_rate * d_b3
        self.weights2 -= learning_rate * d_w2
        self.bias2 -= learning_rate * d_b2
        self.weights1 -= learning_rate * d_w1
        self.bias1 -= learning_rate * d_b1

    def train(self, X, y, epochs=100, learning_rate=0.001):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute the loss
            loss = cross_entropy_loss(predictions, y)
            
            # Backward pass and update
            self.backward(X, y, learning_rate)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        logits = self.forward(X)
        predictions = np.argmax(softmax(logits), axis=1)
        return predictions
