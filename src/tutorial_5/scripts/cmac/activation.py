#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import matplotlib.pyplot as plt

class Activation(object):

    def __init__(self, name):
        self.name = name

    @staticmethod
    def compute_result(data):
        raise NotImplementedError

    @staticmethod
    def compute_gradient(data):
        raise NotImplementedError

    @staticmethod
    def compute_gradient_result(data):
        raise NotImplementedError

    def __str__(self):
        return self.name

class Sigmoid(Activation):

    def __init__(self):
        super(Sigmoid, self).__init__("sigmoid")

    @staticmethod
    def compute_result(data):
        return 1 / (1 + np.exp(-data))

    @staticmethod
    def compute_gradient(data):
        return np.exp(data) / (1 + np.exp(data)) ** 2

    @staticmethod
    def compute_gradient_result(data):
        y = 1 / (1 + np.exp(-data))
        dy = y * (1 - y)
        return y, dy
    

class Tanh(Activation):

    def __init__(self):
        super(Tanh, self).__init__("tanh")

    @staticmethod
    def compute_result(data):
        return np.tanh(data)
    
    @staticmethod
    def compute_gradient(data):
        return 1 - np.tanh(data) ** 2

    @staticmethod
    def compute_gradient_result(data):
        y = np.tanh(data)
        dy = 1 - y ** 2
        return y, dy


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__("ReLU")

    @staticmethod
    def compute_result(data):
        return np.maximum(0, data)

    @staticmethod
    def compute_gradient(data):
        dy = data.copy()
        dy[dy <= 0] = 0
        dy[dy > 0] = 1
        return dy

    @staticmethod
    def compute_gradient_result(data):
        y = np.maximum(0, data)
        dy = data.copy()
        dy[dy <= 0] = 0
        dy[dy > 0] = 1
        return y, dy


class Softmax(Activation):

    def __init__(self):
        super(Softmax, self).__init__("softmax")

    @staticmethod
    def compute_result(data):
        temp = data - np.max(data, axis=0)
        return np.exp(temp) / np.sum(np.exp(temp), axis=0)

    @staticmethod
    def compute_gradient(data):
        # not works for multiple data
        y = self.compute_result(data)
        if data.shape[0] > 1 and data.shape[1] > 1:
            jacob_y = y
        else:
            jacob_y = np.diag(y) - np.outer(y, y)
        return jacob_y

    @staticmethod
    def compute_gradient_result(data):
        # not works for multiple data
        y = self.compute_result(data)
        if data.shape[0] > 1 and data.shape[1] > 1:
            jacob_y = y
        else:
            jacob_y = np.diag(y) - np.outer(y, y)
        return y, jacob_y

class NoActivation(Activation):

    def __init__(self):
        super(NoActivation, self).__init__("none")

    @staticmethod
    def compute_result(data):
        return data

    @staticmethod
    def compute_gradient(data):
        dy = np.ones(data.shape)
        return dy

    @staticmethod
    def compute_gradient_result(data):
        y = data
        dy = np.ones(data.shape)
        return y, dy


if __name__ == "__main__":
    diag_index = np.diag_indices(50)
    activation = Sigmoid
    print(activation)
    data = np.linspace(-5, 5, 50)
    result = activation.compute_result(data)
    gradient = activation.compute_gradient(data)
    plt.figure()
    plt.plot(data, result, data, gradient)
    plt.title("Sigmoid")

    activation = Tanh
    data = np.linspace(-5, 5, 50)
    result = activation.compute_result(data)
    gradient = activation.compute_gradient(data)
    plt.figure()
    plt.plot(data, result, data, gradient)
    plt.title("Tanh")

    activation = ReLU
    data = np.linspace(-5, 5, 50)
    result = activation.compute_result(data)
    gradient = activation.compute_gradient(data)
    plt.figure()
    plt.plot(data, result, data, gradient)
    plt.title("ReLU")

    activation = NoActivation
    data = np.linspace(-5, 5, 50)
    result = activation.compute_result(data)
    gradient = activation.compute_gradient(data)
    plt.figure()
    plt.plot(data, result, data, gradient)
    plt.title("None")

    plt.show()
    
    