"""
Multi Layer Perceptron - only built with a single hidden layer

An assignment of Connectionist Computing - UCD COMP30230 Module

- Support learning the XOR problem
- Support learning the Sin function
- Support identifying handwritten letters trained by the data set below
    Link => http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
"""

__version__ = '0.1'
__author__ = 'Qirun Chen - Student No. 16212138'
__date__ = '27 Apr 2018'

import numpy as np


class MLP:
    def __init__(self, n_i, n_h, n_o, activation='sigmoid', max_epochs=5000,
                 learning_rate=0.1, verbose=2):
        """ Initialize the whole network
        :param n_i: The number of units in the input layer
        :param n_h: The number of units in the hidden layer
        :param n_o: The number of units in the output layer
        :param activation: The activation function for activating the weights and inputs
        :param max_epochs: The maximum iterations to training the MLP, to adjust weights
        :param learning_rate: Indicating how much the delta on weights would be accepted
        :param verbose: How much details should be printed during the training
        """

        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.verbose = np.power(10, verbose-1)
        # Initialize the activation function and its derivative functions
        self.activation, self.d_activation = \
            self.__initialize_activation_func(activation)

        # Initialize the loss function
        self.__initialize_loss_func()

        # Initialize the units to hold activation results
        self.input = np.ones(self.n_i + 1)  # 1 for bias
        self.h = np.ones(self.n_h)
        self.o = np.ones(self.n_o)

        # Initialize the weights of the lower layer and the upper layer
        self.W1, self.W2 = self.__initialize_weights()

    def __initialize_activation_func(self, activation):
        """ Initialize the activation function and its derivative function
        :param activation: The symbol of activation, taking a string
        :return: The assigned activation function and its derivative
        """
        # Temporarily using tanh - a rescaled logistic sigmoid function
        return tanh, d_tanh

    def __initialize_loss_func(self):
        """ Initialize the loss function
        - when n_o > 1, classification should apply cross entropy
        - when n_o = 1, like a regression problem, should use squared error
        :return: None
        """
        if self.n_o > 1:
            self.loss_func = cross_entropy
        else:
            self.loss_func = squared_error

    def __initialize_weights(self):
        """ Randomly initialize weights between -1 and 1
                for the lower layer and the upper layer
        :return: The initialized weights
        """
        w1 = np.random.uniform(-0.2, 0.2, (self.input.size, self.h.size))
        w2 = np.random.uniform(-0.2, 0.2, (self.h.size, self.o.size))
        return w1, w2

    def __forwards(self, inputs):
        """ Propagate the inputs forward from the input layer
                to the output layer
        :param inputs: The training example
        :return: The output on output units
        """

        # In the input layer, the last one is 1 for bias
        self.input[:-1] = inputs
        # Activate the hidden layer
        self.h = self.activation(np.dot(self.input, self.W1))
        if self.n_o > 1:
            # Classification - using SoftMax
            self.o = softmax(np.dot(self.h, self.W2))
        else:
            # Regression
            self.o = self.activation(np.dot(self.h, self.W2))

        return self.o

    def __backwards(self, expected):
        """ The core training technique.
                Propagate the error signal back to each layer,
                and adjust the weights
        :param expected: The expected outputs for calculating the error
        :return: None
        """

        # The error on the output layer
        error = expected - self.o
        # Computing the delta activation of the output layer
        if self.n_o > 1:
            # Classification
            dz2 = error * d_softmax(self.o, softmax)
        else:
            # Regression
            dz2 = error * self.d_activation(self.o)
        # Computing the delta activation of the hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self.d_activation(self.h)

        # Update the weights
        self.__update_weights(dz1, dz2)

    def __update_weights(self, dz1, dz2):
        """ Update weights on the lower layer and the upper layer
        :param dz1: The delta on the hidden layer
        :param dz2: The delta on the output layer
        :return: None
        """
        dw1 = np.dot(np.atleast_2d(self.input).T, np.atleast_2d(dz1))
        self.W1 += self.learning_rate * dw1
        dw2 = np.dot(np.atleast_2d(self.h).T, np.atleast_2d(dz2))
        self.W2 += self.learning_rate * dw2

    def fit(self, X, y):
        """ Training the MLP network by the training set
        Adjust the weights by iterate the training set max-epoch times
        :param X: The features of the training set
        :param y: The labels/Outputs of the training set
        :return: self
        """
        for e in range(1, self.max_epochs):
            loss = 0.
            for j, row in enumerate(X):
                # feed-forward the inputs to the output layer
                o = self.__forwards(row)
                # Accumulate the error of each example computed
                #   by the loss function to get the total loss
                loss += self.loss_func(o, y[j])
                # Back-propagate the error signal computed
                #   according to the given expected output
                self.__backwards(y[j])

            # Print details during training
            if self.n_o > 1:
                # Classification
                # Predict the training set and print the current accuracy
                pre = self.predict(X)
                acc = 0.
                for k, _ in enumerate(y):
                    if pre[k] == np.argmax(y[k]):
                        acc += 1
                if e % self.verbose == 0:
                    print('epoch %d | cost : %.3f | accuracy : %.3f' %
                          (e, loss/len(X), acc/len(X)))
            else:
                if e % self.verbose == 0:
                    # Regression - print the cost (average loss on training dataset)
                    print('epoch %d | cost : %.3f' % (e, loss/(len(X))))

        return self

    def predict(self, X):
        """ Predict on the test set
        :param X: The unseen features of the test set
        :return: The predicted output for each example in the test set
        """
        y = list()
        for j, row in enumerate(X):
            if self.n_o > 1:
                # Classification - using one hot encoding,
                #   so find the index of output units with the max output
                y.append(np.argmax(self.__forwards(row)))
            else:
                y.append(self.__forwards(row))
        return np.array(y)


# Definitions of sigmoid, derivatives, SoftMax, and loss functions
def tanh(x):
    """ The rescaled logistic sigmoid function
    :param x: The input needs to be activated
    :return: corresponding tanh value with the input x
    """
    return np.tanh(x)


def d_tanh(x):
    """ The derivative of the tanh activation function
    :param x: The input value
    :return: The derivative
    """
    return 1.0 - x**2


def softmax(x):
    """ Compute the SoftMax of vector x in a numerically stable way,
            since numpy's exp would lead to infinite (nan)
    :param x: In Classification, the inputs on the output units need to be activated
    :return: The probabilities on each output unit. The sum should be 1.
    """
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def d_softmax(o, f):
    """ The derivative of SoftMax with respect to the output
    :param o: The output
    :param f: The activation function of the output layer - SoftMax in classification
    :return: The derivative
    """
    return f(o) * (1 - f(o))


def cross_entropy(o, y):
    """ Cross entropy loss function
    :param o: The output
    :param y: The expected output
    :return: The cross entropy - the loss
    """
    return np.sum(np.nan_to_num(-y * np.log(o) - (1-y) * np.log(1-o)))


def squared_error(o, y):
    """ Squared error loss function
    :param o: The output
    :param y: The expected output
    :return: The square error - the loss
    """
    return 0.5 * ((y-o) ** 2).sum()


