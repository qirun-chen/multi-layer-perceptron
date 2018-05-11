"""
A test for Multi Layer Perceptron

An assignment of Connectionist Computing - UCD COMP30230 Module

- Test 2 to see if mlp learns the Sin function
"""

__version__ = '0.1'
__author__ = 'Qirun Chen - Student No. 16212138'
__date__ = '27 Apr 2018'

import numpy as np
from scripts.multi_layer_perceptron import MLP
import matplotlib.pyplot as plt


def generate_by_sin(sample_num):
    # Randomly initialize the inputs
    X = np.random.uniform(-1, 1, (sample_num, 4))
    # Compute the output according to the function sin(x1-x2+x3-x4)
    y = list(map(lambda a: np.sin(a[0] - a[1] + a[2] - a[3]), X))
    # Stack the output to the x values
    return X, y


def create_dataset():
    # Initialize the training set and test set
    sample_num = 50
    split_ratio = 0.8
    train_num = int(sample_num * split_ratio)
    X, y = generate_by_sin(sample_num)
    ds = np.column_stack((X, y))
    # Split the training set and test set
    return ds[:train_num], ds[train_num:], X, y


def initialize_network():
    # Initialize the MLP network
    return MLP(4, 10, 1, learning_rate=0.1, max_epochs=3000)


def train_network():
    # Train the MLP network
    mlp.fit(train[:, :-1], train[:, -1])
    return mlp


def predict():
    test_x = test[:, :-1]
    test_y = test[:, -1]
    # Predict on the test set
    prediction = mlp.predict(test_x).flatten()

    # Print out the output with the expected
    cost = 0.
    for i, l in enumerate(test_x):
        cost += 0.5 * (test_y[i] - prediction[i]) ** 2
        print('%s | expected : %.f | output : %.3f' % (str(l), test_y[i], prediction[i]))
    print('error on test set : %.6f' % (cost/len(test_x)))


def plot():
    # Compute the x value for the sin function for plotting
    x = list(map(lambda a: a[0] - a[1] + a[2] - a[3], X))
    output_y = mlp.predict(X)
    # Indicate the figure size
    plt.figure(figsize=(8, 4))
    # Draw the real function and the MLP prediction approximated function
    plt.plot(x, y, 'g^', x, output_y, 'ro')
    # X axis [0, 3] | Y axis [0, 1.3]
    plt.axis([0, 3, 0, 1.3])
    plt.show()


if __name__ == "__main__":
    train, test, X, y = create_dataset()
    mlp = initialize_network()
    mlp = train_network()
    predict()
    plot()


