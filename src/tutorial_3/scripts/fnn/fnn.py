#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import matplotlib.pyplot as plt
from activation import Sigmoid, Tanh, ReLU, Softmax, NoActivation
from initializer import RandnInitializer, RandInitializer, ZeroInitializer

eps = 1e-16

class FNNLayer(object):

    def __init__(self, input_node, output_node, activation_func, w_initializer, b_initializer, regularizer, lambd=0.1, name="hidden"):
        self.name = name # name for __str__
        self.input_node = input_node # number of input node
        self.output_node = output_node # number of output node
        self.activation = activation_func # activation function, class Activation
        self.w_initializer = w_initializer # Weight initializer, class Initializer
        self.b_initializer = b_initializer # Bias initializer, class Initializer
        self.regularizer = regularizer # 2: only implemented L2 Regularization
        self.lambd = lambd # lambda for regularization
        self.w = np.hstack([w_initializer.generate(output_node, input_node), b_initializer.generate(output_node, 1)]) # output x (input + 1)

    # use for training, return output and activation function gradient
    def forward_verbose(self, data):
        # append 1 to input as the bias input
        if len(data.shape) == 1:
            x = np.vstack([data, 1])
        else:
            x = np.vstack([data, [1] * np.size(data, 1)])
        # Wx + b
        z = np.dot(self.w, x)
        # sigma(Wx + b) and d_sigma
        h, dh = self.activation.compute_gradient_result(z)
        # append a zero row to J_sigma, ensure matrix dimension match in gradient descent 
        # jacob_activ = np.vstack([jacob_activ, [0] * self.output_node])
        return h, dh

    # use for prediction
    def forward(self, data):
        # append 1 to input as the bias input
        if len(data.shape) == 1:
            x = np.vstack([data, 1])
        else:
            x = np.vstack([data, [1] * np.size(data, 1)])
        # sigma(Wx + b)
        h = self.activation.compute_result(np.dot(self.w, x))
        return h

    def gradient_descent(self, gradient, lr, batch_size):
        if self.regularizer == 2:
            # W <-- (1 - alpha * lambda) * W - alpha * Gradient_W
            self.w = (1 - lr * self.lambd / batch_size) * self.w  - lr * gradient / batch_size
        else:
            self.w = self.w  - lr * gradient / batch_size

    def __str__(self):
        # Print layer info
        description = self.name + " layer with %d input nodes and %d output nodes, activation: "%(self.input_node, self.output_node) + \
            self.activation.__str__() + ", weight initializer: " + self.w_initializer.__name__ + \
            ", bias initializer: " + self.b_initializer.__name__
        return description


class FNN(object):

    @property
    def Sigmoid(self):
        return Sigmoid

    @property
    def Tanh(self):
        return Tanh
    
    @property
    def ReLU(self):
        return ReLU

    @property
    def Softmax(self):
        return Softmax

    @property
    def NoActivation(self):
        return NoActivation

    @property
    def RandnInitializer(self):
        return RandnInitializer

    @property
    def ZeroInitializer(self):
        return ZeroInitializer

    @property
    def L2Regularization(self):
        return 2

    # label: label vector, n: number of classes
    def one_hot_encode(self, label, n):
        return np.eye(n)[np.array(label, dtype=np.int32).reshape(-1)].T

    def __init__(self, discrete=False, learning_rate=0.01, batch_size=32, loss=None):
        self.layer_list = [] # all layers
        self.discrete = discrete # True: classification, False: Regression
        self.lr = learning_rate # alpha
        self.batch_size = batch_size # m
        self.layer_num = 0 # number of layers
        # select loss function, if output layer is softmax, should use cross entropy, otherwise will cause error
        if loss is None:
            if discrete:
                self.loss_func = 1 # cross entropy
            else:
                self.loss_func = 0 # MSE
        else:
            self.loss_func = loss

    # add a layer to the neural network, no validation check, please only use softmax for output layer
    def add_layer(self, input_node, output_node, activation, w_initializer, b_initializer, regularizer=2, lambd=0.1, name="hidden"):
        activ_func = activation()
        new_layer = FNNLayer(input_node, output_node, activ_func, w_initializer, b_initializer, regularizer, lambd, name)
        self.layer_list.append(new_layer)
        self.layer_num += 1

    # predict without gradient
    def predict(self, data):
        x = data.copy()
        for layer in self.layer_list:
            y = layer.forward(x)
            x = y
        return y

    # compute loss function
    def compute_loss(self, y_pred, y):
        # MSE
        if self.loss_func == 0:
            loss = np.sum((y_pred - y) ** 2, axis=0) / 2
        # Cross Entropy
        elif self.loss_func == 1:
            # eps to avoid math error
            loss = -np.sum(y * np.log(y_pred + eps), axis=0)
        return loss
    
    # apply batch training to the full dataset, N data, x_set: in x N, y_set: out x N
    def train(self, x_set, y_set):
        N = x_set.shape[1] # N
        loss_trace = [] # store loss trace
        # select data in order
        for i_start in range(0, N, self.batch_size):
            print(i_start)
            current_batch_size = min(i_start + self.batch_size, N) - i_start
            x_batch = x_set[:, i_start:(i_start + current_batch_size)]
            y_batch = y_set[:, i_start:(i_start + current_batch_size)]
            # batch training
            loss = self.train_batch(x_batch, y_batch)
            loss_trace.append(np.mean(loss))
        return loss_trace         

    # batch training, N data in batch, update only once
    def train_batch(self, x_batch, y_batch):
        N = x_batch.shape[1]
        # if only one data, reshape to a vector
        if N == 1:
            x_batch = x_batch.reshape((-1, 1))
            y_batch = y_batch.reshape((-1, 1))
        dloss, dh, h, loss = self.compute_gradient(x_batch, y_batch)      
        self.gradient_descent(dloss, dh, h, N)
        return loss

    # only train for one step
    def train_one_step(self, x, y):
        dloss, dh_list, h_list, loss = self.compute_gradient(x, y)
        self.gradient_descent(dloss, dh_list, h_list, 1)        
        return loss

    def compute_gradient(self, x, y):
        N = x.shape[1] # number of data pairs
        h_list = []
        dh_list = []
        # z_l = W_l*h_l-1 + b_l
        # h_l = sigma(z_l), layer output, next layer input
        h = x.copy()
        for layer in self.layer_list:
            h_homo = np.vstack([h, [1] * N])
            h_list.append(h_homo)
            h, dh = layer.forward_verbose(h)
            dh_list.append(dh)

        loss = self.compute_loss(h, y)

        # MSE loss function gradient
        if self.loss_func == 0: 
            dloss = h - y
        # cross entropy loss function gradient with softmax output, easy computation
        elif self.loss_func == 1 and self.layer_list[-1].activation.name == "softmax":
            dloss = 1
            error = h - y
            dh_list[-1] = error
        # cross entropy with other output activation function, can cause divide by zero exception!
        elif self.loss_func == 1:
            dloss = -y / h
            dloss = dloss.T
        # no idea
        else:
            dloss = 1

        return dloss, dh_list, h_list, loss
        
    def generate_wx_jacobian(self, x, m):
        return np.kron(np.eye(m), x.flatten())

    # Compute gradient for every layer from Jacobians and apply gradient descent once
    def gradient_descent(self, dloss, dh_list, h_list, batch_size=1):
        w_list = [] # W in all layers
        for layer_index in range(1, self.layer_num):
            w_list.append(self.layer_list[layer_index].w[:, :-1])
        # output layer L, dL: loss gradient
        # gradient_output dL/dW_L = dL/dh_L * dh_L/dz_L * dz_L/dW_L
        #                         = special loss and output activation gradient * h_L-1.T
        # print(dloss.shape)
        # print(dh_list[-1].shape)
        jacobian = dloss * dh_list[-1] 
        # print(jacobian.shape)
        # print(h_list[-1].shape)
        dW_L = np.dot(jacobian, h_list[-1].T)
        self.layer_list[-1].gradient_descent(dW_L, self.lr, batch_size)        

        # iteration from the last hidden layer        
        # dL/dh_l-1 = Jacobian * dz_l/dh_L-1 = Jacobian * W_l
        # dL/dz_l-1 = dL/dh_l-1 * dh_l-1/dz_l-1 = dL/dh_l-1 * activation gradient
        # dL/dW_l-1 = dL/dz_l-1 * dz_l-1/dW_l-1 = dL/dz_l-1 * h_l-2.T
        for layer_index in range(2, self.layer_num + 1):
            jacobian = np.dot(w_list[-layer_index + 1].T, jacobian)
            jacobian = jacobian * dh_list[-layer_index]
            dW_l = np.dot(jacobian, h_list[-layer_index].T)
            self.layer_list[-layer_index].gradient_descent(dW_l, self.lr, batch_size)

    def save_parameter(self, filename):
        for i in range(len(self.layer_list)):
            np.savetxt(filename + "_" + str(i) + ".txt", self.layer_list[i].w)
        print("Save to " + filename)

    def load_parameter(self, filename):
        for i in range(len(self.layer_list)):
            self.layer_list[i].w = np.loadtxt(filename + "_" + str(i) + ".txt", delimiter=" ")
        print("Load parameter from " + filename)
    
    def __str__(self):
        description = "Feedforward Neural Network with %d layers: \n" % (self.layer_num)
        for layer in self.layer_list:
            description += layer.__str__() + "\n"
        return description


if __name__ == "__main__":
    nn = FNN(discrete=True, learning_rate=0.01)
    nn.add_layer(5, 20, nn.ReLU, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.1, "hidden1")
    nn.add_layer(20, 20, nn.ReLU, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.1, "hidden2")
    nn.add_layer(20, 3, nn.Softmax, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.1, "output")
    print(nn)

    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) / 5
    print(data)
    y_pred = nn.predict(data)
    print(y_pred)    
    y = np.array([0, 1, 0]).reshape(-1, 1)
    print(nn.compute_loss(y_pred, y))

    for i in range(0, 10):
        nn.train_one_step(data, y)
    y_pred = nn.predict(data)
    print(y_pred)
    print(nn.compute_loss(y_pred, y))
    



    
