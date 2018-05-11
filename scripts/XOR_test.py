"""
A test for Multi Layer Perceptron

An assignment of Connectionist Computing - UCD COMP30230 Module

- Test 1 to see if mlp learns the XOR problem
"""

__version__ = '0.1'
__author__ = 'Qirun Chen - Student No. 16212138'
__date__ = '27 Apr 2018'

import numpy as np
from scripts.multi_layer_perceptron import MLP

if __name__ == "__main__":

    # Initialize the XOR inputs
    XOR_inputs = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    # Split the inputs and outputs
    x = XOR_inputs[:, :-1]
    y = XOR_inputs[:, -1]

    # Initialize the MLP network
    mlp = MLP(2, 2, 1, max_epochs=2000)

    # Training the MLP
    mlp.fit(x, y)

    # Predict the MLP on XOR inputs
    prediction = mlp.predict(x)
    for i, l in enumerate(y):
        print('%s | expected : %.f | output : %.3f' % (str(x[i]), y[i], prediction[i]))


