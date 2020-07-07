#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI


import numpy as np
import matplotlib.pyplot as plt
from activation import Sigmoid, Tanh, ReLU, Softmax, NoActivation
from initializer import RandnInitializer, RandInitializer, ZeroInitializer


class CMAC(object):

    def __init__(self, input_number, output_number, receptive_field, resolution, initializer=RandnInitializer, activation_func=NoActivation, lr=0.1):
        self.input_number = input_number
        self.output_number = output_number
        self.receptive_field = receptive_field
        self.resolution = resolution
        self.initializer = initializer
        self.activation_func = activation_func
        self.lr = lr
        if self.receptive_field % 2 == 0:
            self.receptive_before = self.receptive_field / 2
            self.receptive_after = self.receptive_before
        else:
            self.receptive_before = np.floor(self.receptive_field / 2)
            self.receptive_after = self.receptive_before + 1
        
        self.neuron_position_mask = self.generate_position_matrix()
        self.neuron_number = np.sum(self.neuron_position_mask)
        self.neuron_matrix_init = np.zeros((self.resolution, self.resolution))

        self.weight_matrix = self.initializer.generate(output_number, self.neuron_number)
        
    # only for two input
    def generate_position_matrix(self):
        offset_list = [[], [0], [1, 0], [2, 0, 1], [2, 0, 1, 3], [3, 0, 2, 4, 1], [4, 0, 2, 5, 1, 3], [5, 0, 4, 1, 3, 6, 2]]
        position_matrix = np.zeros((self.resolution, self.resolution))
        offset = offset_list[self.receptive_field]
        for row in range(self.resolution):
            index = row % self.receptive_field
            position_matrix[row, 0 + offset[index]::self.receptive_field] = 1
        return position_matrix

    
    # def generate_random_neuron_matrix(self, neuron_num):
    #     # generate a matrix with 1 at random position
    #     neuron_array = np.zeros((self.resolution * self.resolution))
    #     neuron_array[:neuron_num] = 1
    #     neuron_matrix = neuron_array.reshape((self.resolution, self.resolution))
    #     while not np.all(np.sum(neuron_matrix, axis=0)) or not np.all(np.sum(neuron_matrix, axis=1))
    #         np.random.shuffle(neuron_array)
    #         neuron_matrix = neuron_array.reshape((self.resolution, self.resolution))
    #     return neuron_matrix

    # only for two input
    def predict(self, data, verbose=False):
        activ_center = np.round(data * self.resolution)
        neuron_activ_matrix = self.activate_neuron(activ_center)
        neuron_activ_vec = self.neuron_mat2vec(neuron_activ_matrix)
        z = np.dot(self.weight_matrix, neuron_activ_vec)
        output = self.activation_func.compute_result(z)
        if verbose:
            return output, neuron_activ_vec
        else:
            return output

    def train(self, x_dataset, y_dataset):
        N = x_dataset.shape[1]

        loss_trace = []
        
        for i in range(N):
            x = x_dataset[:, i]
            y = y_dataset[:, i]
            y_pred, neuron_activ_vec = self.predict(x, verbose=True)
            error = y - y_pred
            update_position = np.concatenate([neuron_activ_vec.T, neuron_activ_vec.T], axis=0)
            self.weight_matrix = self.weight_matrix + self.lr / self.receptive_field * update_position * error
            loss_trace.append(self.compute_loss(y, y_pred))
        return np.array(loss_trace)

    def train_one_step(self, x, y):
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        y_pred, neuron_activ_vec = self.predict(x, verbose=True)
        error = y - y_pred
        update_position = np.concatenate([neuron_activ_vec.T, neuron_activ_vec.T], axis=0)
        self.weight_matrix = self.weight_matrix + self.lr / self.receptive_field * update_position * error
        return self.compute_loss(y, y_pred)

    def compute_loss(self, y, y_pred):
        return np.sum((y - y_pred) ** 2) / 2


    def neuron_mat2vec(self, mat=None):
        if mat is None:
            mat = self.neuron_matrix
        neuron_vec = mat[self.neuron_position_mask > 0]
        return neuron_vec.reshape(-1, 1)

    # only for two input
    def activate_neuron(self, center):
        slice_row = slice(int(center[0] - self.receptive_before), int(center[0] + self.receptive_after))
        slice_col = slice(int(center[1] - self.receptive_before), int(center[1] + self.receptive_after))
        activ_mat = self.neuron_matrix_init.copy()
        activ_mat[slice_row, slice_col] = 1
        return activ_mat

    def save_parameter(self, filename):
        np.savetxt(filename + ".txt", self.weight_matrix)
        print("Save to " + filename + ".txt")

    def load_parameter(self, filename):
        self.weight_matrix = np.loadtxt(filename + ".txt", delimiter=" ")
        print("Load parameter from " + filename + ".txt")

    def __str__(self):
        output = "CMAC, output node %d, resolution %d, receptive field %d, L2 neuron number %d" % (self.output_number, self.resolution, self.receptive_field, self.neuron_number)
        return output     



if __name__ == "__main__":
    cmac_test = CMAC(2, 2, 5, 50, RandnInitializer, NoActivation)
    print(cmac_test)
    for i in range(5):
        print(cmac_test.predict(np.random.rand(2)))





