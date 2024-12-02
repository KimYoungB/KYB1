from cProfile import label
from statistics import correlation


import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:

    def __init__(self, n_inputs, n_outputs):
        '''
        Args:
        n_inputs:
        n_outputs:
        '''
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))

        def forward(self,inputs):
            '''
            Args:
              inputs:
            '''
            self.inputs = inputs
            self.outputs = np.dot(inputs, self.weights) + self.biases


        def backward(self, dvalues):
            '''
            Args:
                dvalues:
            '''
            # f(x,w) = xw
            # w 편미분 f'(x,w) = x
            self.dweights = np.dot(self.inputs.T, dvalues)
            # f(x,y) = x + y
            # y 편미분 = 1
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
            self.dweights = np.dot(dvalues.T, self.inputs.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
        '''
        Args:
            inputs:
        '''
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.outputs = probabilities

    def backward(self, dvalues):
        '''
        Args:
            dvalues:
        '''
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, self.dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobin_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobin_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = len(y_pred)
        y_pred_clipped = np.clip(y_pred, a_min: 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(sample_losses), y_true
            ]
            elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

            negative_log_likehoods = - np.log(correct_confidences)
            return negative_log_likehoods

    def backward(self, dvalues, y_turn):
        sample = len(dvalues)
        labbels__ len(dvalues[0])

        if len(y_turn.shape) == 1:
            y_turn = np.eye(labels)[y_turn]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples

class Activation_SoftmaxLoss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, y_pred, y_true):
        self.activation.forward(y_pred)
        self.outputs = self.activation.outputs
        return self.loss.forward(y_true, self.outputs)

    def backward(self, dvalues, y_turn):
        samples = len(dvalues)

        if len(y_turn.shape) == 2:
            y_ture = np.argmax(y_turn, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_ture] -= 1
        self.dinputs = self.dinputs / samples

    dense1 = Layer_Dense(2, 1)
    activtion1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    loss_activation = Activation_SoftmaxLoss_CategoricalCrossentropy


X , y spiral_data(samples=100, classes=3)

dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = loss_activation.forward(dense2.output, y)

print('loss', loss)

predictions = np.argmax(loss_activaiton.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print('accuracy', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
