import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):

        self.weights = np.random.uniform(low=0, high=1, size=(n_inputs, n_neurons))

        self.biases = np.random.uniform(low=0, high=1, size=(1, n_neurons))

    def forward(self, inputs):

        return np.dot(inputs, self.weights) + self.biases

inputs = np.array([[1, 2], [3, 4], [5, 6]])
layer = Layer_Dense(2, 3)

output = layer.forward(inputs)
print("Output:")
print(output)

# 실습 3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None, init_method="he"):

        self.weights = self.initialize_weights(n_inputs, n_neurons, method=init_method)

        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def initialize_weights(self, n_inputs, n_neurons, method="he"):
        if method == "random":
            return np.random.randn(n_inputs, n_neurons) * 0.01
        elif method == "xavier":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif method == "he":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError("Unsupported initialization method")

    def forward(self, inputs):

        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation:
            return self.activation(self.z)
        return self.z

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X, Y = make_moons(n_samples=100, noise=0.2, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="bwr")
plt.title("Non-linear Data (Moons)")
plt.show()

input_size = 2
hidden_size = 5
output_size = 1

hidden_layer = DenseLayer(input_size, hidden_size, activation=relu, init_method="he")

output_layer = DenseLayer(hidden_size, output_size, activation=sigmoid, init_method="xavier")

hidden_output = hidden_layer.forward(X)

predictions = output_layer.forward(hidden_output)

print("Predictions (first 5 samples):")
print(predictions[:5])

plt.scatter(X[:, 0], X[:, 1], c=predictions.reshape(-1), cmap="bwr", marker='o')
plt.title("Predictions Visualization")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

)
class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None, init_method="he"):

        self.weights = self.initialize_weights(n_inputs, n_neurons, method=init_method)

        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def initialize_weights(self, n_inputs, n_neurons, method="he"):
        if method == "random":
            return np.random.randn(n_inputs, n_neurons) * 0.01
        elif method == "xavier":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif method == "he":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError("Unsupported initialization method")

    def forward(self, inputs):

        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation:
            return self.activation(self.z)
        return self.z

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X, Y = make_moons(n_samples=100, noise=0.2, random_state=42)

input_size = 2
hidden_size_2 = 3
output_size = 5

hidden_layer_1 = DenseLayer(input_size, hidden_size_1, activation=relu, init_method="he")

hidden_layer_2 = DenseLayer(hidden_size_1, hidden_size_2, activation=relu, init_method="he")

output_layer = DenseLayer(hidden_size_2, output_size, activation=sigmoid, init_method="xavier")

hidden_output_1 = hidden_layer_1.forward(X)

hidden_output_2 = hidden_layer_2.forward(hidden_output_1)

final_output = output_layer.forward(hidden_output_2)

print("Final Output (first 5 samples):")
print(final_output[:5])


# 출력값 제한하기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None, init_method="he"):

        self.weights = self.initialize_weights(n_inputs, n_neurons, method=init_method)

        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def initialize_weights(self, n_inputs, n_neurons, method="he"):
        if method == "random":
            return np.random.randn(n_inputs, n_neurons) * 0.01
        elif method == "xavier":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif method == "he":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError("Unsupported initialization method")

    def forward(self, inputs):

        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation:
            return self.activation(self.z)
        return self.z

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def limit_output(x):

    return np.maximum(0, x)

X, Y = make_moons(n_samples=100, noise=0.2, random_state=42)

input_size = 2
hidden_size_1 = 3
hidden_size_2 = 3
output_size = 5

hidden_layer_1 = DenseLayer(input_size, hidden_size_1, activation=relu, init_method="he")

hidden_layer_2 = DenseLayer(hidden_size_1, hidden_size_2, activation=relu, init_method="he")

output_layer = DenseLayer(hidden_size_2, output_size, activation=sigmoid, init_method="xavier")

hidden_output_1 = hidden_layer_1.forward(X)

hidden_output_2 = hidden_layer_2.forward(hidden_output_1)

final_output = output_layer.forward(hidden_output_2)

limited_output = limit_output(final_output)

print("Limited Output (first 5 samples):")
print(limited_output[:5])

plt.scatter(X[:, 0], X[:, 1], c=limited_output[:, 0], cmap="bwr", marker='o')
plt.title("Predictions Visualization for First Output Neuron (Limited Output)")
plt.show()


#3 가중치 초기화 방법 실험하기

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None, initialize_method="he"):
        # 가중치 초기화 (새로운 초기화 방식 추가)
        self.weights = self.initialize_weights(n_inputs, n_neurons, method=initialize_method)
        # 편향을 0으로 초기화
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def initialize_weights(self, n_inputs, n_neurons, method="he"):
        if method == "random":
            return np.random.randn(n_inputs, n_neurons) * 0.01
        elif method == "xavier":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif method == "he":
            return np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

    def forward(self, inputs):
        # 입력 * 가중치 + 편향
        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation:
            return self.activation(self.z)
        return self.z

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def limit_output(x):
    return np.maximum(0, x)

X, Y = make_moons(n_samples=100, noise=0.2, random_state=42)

def build_and_run_dnn(init_method):
    input_size = 2
    hidden_size_1 = 3
    hidden_size_2 = 3
    output_size = 5

    hidden_layer_1 = DenseLayer(input_size, hidden_size_1, activation=relu, initialize_method=init_method)

    hidden_layer_2 = DenseLayer(hidden_size_1, hidden_size_2, activation=relu, initialize_method=init_method)

    output_layer = DenseLayer(hidden_size_2, output_size, activation=sigmoid, initialize_method=init_method)

    hidden_output_1 = hidden_layer_1.forward(X)
    hidden_output_2 = hidden_layer_2.forward(hidden_output_1)
    final_output = output_layer.forward(hidden_output_2)

    limited_output = limit_output(final_output)

    print(f"Results for initialization method: {init_method}")
    print(limited_output[:5])

    plt.scatter(X[:, 0], X[:, 1], c=limited_output[:, 0], cmap="bwr", marker='o')
    plt.title(f"Predictions for {init_method} Initialization (First Neuron)")
    plt.show()

for method in ["random", "xavier", "he"]:
    build_and_run_dnn(method)