import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


X, y = spiral_data(samples=2, classes=3)
dense1 = Layer_Dense(2, 3)

dense_output = dense1.forward(X)
print("Dense Layer Output:")
print(dense_output)

activation1 = Activation_ReLU()
activation_output = activation1.forward(dense_output)

print("\nActivation (ReLU) Output:")
print(activation_output)

import numpy as np
import matplotlib.pyplot as plt



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output



X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

dense1 = Layer_Dense(1, 8)
activation1 = Activation_Tanh()

dense2 = Layer_Dense(8, 8)
activation2 = Activation_Tanh()

dense3 = Layer_Dense(8, 1)
activation3 = Activation_Tanh()

layer1_output = dense1.forward(X)
activation1_output = activation1.forward(layer1_output)

layer2_output = dense2.forward(activation1_output)
activation2_output = activation2.forward(layer2_output)

layer3_output = dense3.forward(activation2_output)
activation3_output = activation3.forward(layer3_output)

plt.plot(X, y, label="True Sine Wave", color="blue")
plt.plot(X, activation3_output, label="NN output", color="red")
plt.legend()
plt.title("Sine Wave Approximation using Neural Network")
plt.show()

