#!/usr/bin/env python3
""" Deep Neural Network """

import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification.
    """

    def __init__(self, nx, layers):
        """ Instantiation function

        Args:
            nx (int): number of input features
            layers (list): representing the number of nodes in each layer of
                           the network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Return layers in the neural network """
        return self.__L

    @property
    def cache(self):
        """ Return dictionary with intermediate values of the network """
        return self.__cache

    @property
    def weights(self):
        """Return weights and bias dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation

        Args:
            X (numpy.array): Input array with
            shape (nx, m) = (features, no of examples)
        """
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            A_prev = self.cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the Neural Network.

        Args:
            Y (numpy.array): Actual values
            A (numpy.array): predicted values of the neural network

        Returns:
            cost (float): the cost of the predictions
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: Predicted labels and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
