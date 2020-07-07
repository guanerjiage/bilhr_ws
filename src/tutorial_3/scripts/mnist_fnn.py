#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
from fnn.fnn import FNN
from mnist_dataset.mnist import unpack_mnist_label, unpack_mnist_image, download_mnist_dataset

# one hot encoder
def one_hot_encode(label, n):
    return np.eye(n)[np.array(label, dtype=np.int32).reshape(-1)].T

if __name__ == "__main__":
    # download dataset if not exists
    download_mnist_dataset("mnist_dataset/")
    # load training dataset, reshape to required shape
    training_set = unpack_mnist_image("mnist_dataset/train-images.idx3-ubyte")
    training_set_matrix = training_set.reshape((60000, -1)).transpose()
    training_label = unpack_mnist_label("mnist_dataset/train-labels.idx1-ubyte")
    training_label_matrix = one_hot_encode(training_label, 10)
    # load test dataset, reshape to required shape
    test_set = unpack_mnist_image("mnist_dataset/t10k-images.idx3-ubyte")
    test_set_matrix = test_set.reshape((10000, -1)).transpose()
    test_label = unpack_mnist_label("mnist_dataset/t10k-labels.idx1-ubyte")
    test_label_matrix = one_hot_encode(test_label, 10)

    # learning rate
    lr = 0.1
    # initialize neural network, here 2 Sigmoid hidden layer and 1 softmax output layer
    nn = FNN(discrete=True, learning_rate=lr, batch_size=10)
    nn.add_layer(28 * 28, 128, nn.Sigmoid, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden1")
    nn.add_layer(128, 128, nn.Sigmoid, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden2")
    nn.add_layer(128, 10, nn.Softmax, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "output")
    print(nn)

    # select data for training and test
    train_x_set = training_set_matrix[:, 1000:51000]
    print(train_x_set.shape)
    train_y_set = training_label_matrix[:, 1000:51000]
    print(train_y_set.shape)

    test_x_set = test_set_matrix[:, 0:10000]
    test_y_set = test_label_matrix[:, 0:10000]
    test_y_label = test_label[0:10000]

    # normalization
    # Gaussian(0, 1)
    data_mean = np.mean(train_x_set)
    data_std = np.std(train_x_set)
    train_x_set = (train_x_set - data_mean) / data_std
    test_x_set = (test_x_set - data_mean) / data_std

    # linear in range (0, 1)
    # data_max = np.max(train_x_set)
    # data_min = np.min(train_x_set)
    # train_x_set = (train_x_set - data_min) / (data_max - data_min)    
    # test_x_set = (test_x_set - data_min) / (data_max - data_min)    

    loss_trace = []
    acc_valid_trace = []
    
    # update in every step, slow but accuracy high
    lr = 0.1
    nn.lr = lr
    for i in range(50000):
        # learning rate decay
        if i > 30000:
            lr *= 0.999
        nn.lr = lr
        # learn and record loss
        loss_trace.append(nn.train_one_step(train_x_set[:, i].reshape((-1, 1)), train_y_set[:, i].reshape((-1, 1))))
        # print progress
        if i % 1000 == 0:
            print(i)
        # validation
        if i % 50 == 0:            
            test_list = []
            for i in range(200):
                x = test_x_set[:, i].reshape((-1, 1))
                y = test_y_set[:, i].reshape((-1, 1))
                y_pred = nn.predict(x)
                if np.argmax(y_pred) == test_y_label[i]:
                    test_list.append(1)
                else:
                    test_list.append(0)
            acc_valid_trace.append(np.sum(np.array(test_list)) / 200.0 * 100.0)
    plt.figure()
    plt.plot(acc_valid_trace)

    # # batch learning, much more faster, accuracy low
    # lr = 0.4
    # nn.lr = lr
    # loss_trace = nn.train(train_x_set, train_y_set)
    
    # show some prediction results
    for i in range(200, 210):
        x = test_x_set[:, i].reshape((-1, 1))
        y = test_y_set[:, i].reshape((-1, 1))
        y_pred = nn.predict(x)
        print(y_pred.flatten())
        print("%d, %d" %(np.argmax(y_pred), test_y_label[i]))

    # test 
    test_list = []
    loss_list = []
    for i in range(200, 10000):
        x = test_x_set[:, i].reshape((-1, 1))
        y = test_y_set[:, i].reshape((-1, 1))
        y_pred = nn.predict(x)
        if np.argmax(y_pred) == test_y_label[i]:
            test_list.append(1)
        else:
            test_list.append(0)
        loss = nn.compute_loss(y_pred, y)
        loss_list.append(loss)
    print(np.mean(np.array(loss_list)))
    print("Accuracy %.2f%%" % (np.sum(np.array(test_list)) / 9800.0 * 100.0))

    plt.figure()
    plt.plot(loss_trace)
    plt.show()

