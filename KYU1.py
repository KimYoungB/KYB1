 inputs = [1.0, 2.0, 3.0]
 weights = [0.2,0.8,-0.5]
 bias = 2.0

    output = \
        inputs[0]*weights[0] +\
        inputs[1]*weights[1] +\
        inputs[2]*weights[2] +\
        bias

import random

 def init_weights(inputs):
     weights = []
     for i in range(len(inputs)):
         weights.append(random.uniform(-1,b: 1))
         return weights


 def cal(inputs, weights, bias):
     output = sum(i * w for i, w in zip(inputs, weights)) + bias
     return output


import random

def cal_neuron(num_neuron, inputs):
    outputs = []
    for _ in range(num_neuron):
        weights = init_weight(inputs)
        bias = random.uniform(-1, 1)
        output = cal(inputs, weights, bias)
        outputs.append(output)
    return outputs