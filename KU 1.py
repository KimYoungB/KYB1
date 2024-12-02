#1번 문제
import numpy as np

inputs = [[0.5, 1.2, -1.5, 2.0],
          [2.5, -1.0, 0.8, -1.5],
          [-2.0, 3.0, 1.0, 0.5]]

weights = [[-0.1, 0.9, -0.2, 1.0],
           [0.4, -0.7, 0.3, -0.8],
           [-0.3, 0.2, 0.5, 0.6]]

bias = [1.0, 2.0, 3.0]

layers_outputs = np.dot(inputs,np.array(weights).T)+bias
layers_outputs = np.dot(layers_outputs,np.array(weights).T)+bias
print(layers_outputs)

#2번 문제
import numpy as np

inputs = [[0.5, 1.2, -1.5, 2.0],
          [2.5, -1.0, 0.8, -1.5],
          [-2.0, 3.0, 1.0, 0.5],
          [1.0, 2.0, -0.5, 3.5],
          [0.3, -0.5, 2.1, 1.7]]

weights = [[-0.1, 0.9, -0.2, 1.0],
           [0.4, -0.7, 0.3, -0.8],
           [-0.3, 0.2, 0.5, 0.6]]

bias = [1.0, 2.0, 3.0]

layers_outputs = np.dot(inputs,np.array(weights).T)+bias
layers_outputs = np.dot(layers_outputs,np.array(weights).T)+bias
print(layers_outputs)

#3번 문제
import numpy as np


inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3 - 0.8]]

weights_layer1 = [[0.2, 0.8, -0.5, 1.0],
                  [0.5, -0.91, 0.26, -0.5],
                  [-0.26, -0.27, 0.17, 0.87]]

bias_layer1 = [2.0, 3.0, 5.0]

layers1_outputs = np.dot(inputs,np.array(weights_layer1).T)+bias_layer1
layers1_outputs = np.dot(layers1_outputs,np.array(weights).T)+bias_layer1
print(layers_outputs)
print("Layer 1 Output:\n", layers1_outputs)

weights_layer2 = [[0.1, -0.14, 0.5],
                  [-0.5, 0.12, -0.33],
                  [0.3, 0.9, -0.44]]

bias_layer2 = [1.0, 2.0, -0.5]

layers2_outputs = np.dot(inputs,np.array(weights_layer2).T)+bias_layer2
layers2_outputs = np.dot(layers1_outputs, np.array(weights_layer2).T) + bias_layer2
print("Layer 2 Output:\n", layers2_outputs)

