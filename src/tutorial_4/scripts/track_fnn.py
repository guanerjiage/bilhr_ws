#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
from fnn.fnn import FNN

if __name__ == "__main__":
    np.random.seed(233333)

    # load training dataset, reshape to required shape
    training_set = []
    training_label = []
    with open("training_x.txt", "r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines[:-1]:
            im_x = float(line.split(" ")[0])
            im_y = float(line.split(" ")[1])
            training_set.append([im_x, im_y])
    with open("training_y.txt", "r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines[:-1]:
            pitch = float(line.split(" ")[0])
            roll = float(line.split(" ")[1])
            training_label.append([pitch, roll])

    training_set_matrix = np.array(training_set).T
    training_label_matrix = np.array(training_label).T

    # normalization
    # pixel 240 x 320, normalize to [0, 1]
    training_set_matrix[0, :] = training_set_matrix[0, :] / 320.0
    training_set_matrix[1, :] = training_set_matrix[1, :] / 240.0
    # pitch [-2, 2] to [0, 1], roll [0, 1]
    # training_label_matrix[0, :] = (training_label_matrix[0, :] + 2) / 4.0
    # direct output, no normalization

    # load test dataset, reshape to required shape
    test_set_matrix = training_set_matrix[:, np.arange(1, 151, 10)]
    test_label_matrix = training_label_matrix[:, np.arange(1, 151, 10)]

    # learning rate
    lr = 0.05
    # initialize neural network, here 1 ReLU hidden layer and 1 direct output layer
    nn = FNN(discrete=False, learning_rate=lr, batch_size=1)
    nn.add_layer(2, 32, nn.ReLU, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden1")
    # nn.add_layer(64, 64, nn.Sigmoid, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden2")
    nn.add_layer(32, 2, nn.NoActivation, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "output")
    print(nn)


    # select data for training and test
    train_x_set = training_set_matrix.copy()
    train_x_set = np.delete(train_x_set, np.arange(1, 151, 10), axis=1)
    print(train_x_set.shape)
    # print(train_x_set)
    train_y_set = training_label_matrix.copy()
    train_y_set = np.delete(train_y_set, np.arange(1, 151, 10), axis=1)
    print(train_y_set.shape)

    test_x_set = test_set_matrix
    test_y_set = test_label_matrix  
    print(test_x_set.shape)
    print(test_y_set.shape)
    # print(test_x_set)  
    
    loss_epoch = []
    
    # update in every step, slow but accuracy high
    lr = 0.05
    nn.lr = lr
    for i in range(30):
        loss_trace = []
        for j in range(train_x_set.shape[1]):
            loss_trace.append(nn.train_one_step(train_x_set[:, j].reshape((-1, 1)), train_y_set[:, j].reshape((-1, 1))))
        # print progress
        loss_ep = np.mean(loss_trace)
        loss_epoch.append(loss_ep)
        print("Epoch %d, loss %.8f" % (i, loss_ep))

    # test 
    loss_list = []
    for i in range(test_x_set.shape[1]):
        print(i)
        x = test_x_set[:, i].reshape((-1, 1))
        y = test_y_set[:, i].reshape((-1, 1))
        # print(x)
        # print(y)
        y_pred = nn.predict(x)
        loss = nn.compute_loss(y_pred, y)
        loss_list.append(loss)
    print("Test loss %.8f" % np.mean(np.array(loss_list)))

    # save parameters in track_network_i.txt
    nn.save_parameter("track_network")

    plt.figure()
    plt.plot(loss_epoch)
    plt.ylim([-0.1, 4])
    plt.show()
