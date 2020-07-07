
#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
from cmac.cmac import CMAC
from cmac.activation import Sigmoid, Tanh, ReLU, Softmax, NoActivation
from cmac.initializer import RandnInitializer, RandInitializer, ZeroInitializer

if __name__ == "__main__":
    np.random.seed(233333)

    # load training dataset, reshape to required shape
    # training_set = []
    # training_label = []
    # with open("training_x_norm.txt", "r") as f:
    #     data = f.read()
    #     lines = data.split("\n")
    #     for line in lines[:-1]:
    #         im_x = float(line.split(" ")[0])
    #         im_y = float(line.split(" ")[1])
    #         training_set.append([im_x, im_y])
    # with open("training_y_norm.txt", "r") as f:
    #     data = f.read()
    #     lines = data.split("\n")
    #     for line in lines[:-1]:
    #         pitch = float(line.split(" ")[0])
    #         roll = float(line.split(" ")[1])
    #         training_label.append([pitch, roll])

    # training_set_matrix = np.array(training_set).T
    # training_label_matrix = np.array(training_label).T

    training_set_matrix = np.loadtxt("training_x_norm.txt")
    training_label_matrix = np.loadtxt("training_y_norm.txt")

    # use normalized dataset, no need
    # normalization
    # pixel 240 x 320, normalize to [0, 1]
    # training_set_matrix[0, :] = training_set_matrix[0, :] / 320.0
    # training_set_matrix[1, :] = training_set_matrix[1, :] / 240.0
    # pitch [-1.5, 1.5] to [0, 1], roll [0, 1]
    # training_label_matrix[0, :] = (training_label_matrix[0, :] + 1.5) / 3.0
    # direct output, no normalization

    # load test dataset, reshape to required shape
    test_set_matrix = training_set_matrix[:, np.arange(10, 151, 10)]
    test_label_matrix = training_label_matrix[:, np.arange(10, 151, 10)]
    
    # select data for training
    train_x_set = training_set_matrix.copy()
    train_x_set = np.delete(train_x_set, np.arange(10, 151, 10), axis=1)
    # train_x_set = train_x_set[:, ::2]
    print(train_x_set.shape)
    # print(train_x_set)
    train_y_set = training_label_matrix.copy()
    train_y_set = np.delete(train_y_set, np.arange(10, 151, 10), axis=1)
    # train_y_set = train_y_set[:, ::2]
    print(train_y_set.shape)

    # select data for test
    test_x_set = test_set_matrix
    test_y_set = test_label_matrix  
    print(test_x_set.shape)
    print(test_y_set.shape)
    # print(test_x_set) 
    
    # initialize CMAC
    lr = 0.3 # learning rate
    receptive_field = 5   
    cmac_predictor = CMAC(2, 2, receptive_field, 50, lr=lr)
    print(cmac_predictor)

    # loss_trace = []
    loss_epoch = []
    loss_valid_epoch = []
    
    # 50 epoches
    lr = 0.3
    cmac_predictor.lr = lr
    for i in range(50):
        loss_trace = []
        for j in range(train_x_set.shape[1]):
            loss_trace.append(cmac_predictor.train_one_step(train_x_set[:, j].reshape((-1, 1)), train_y_set[:, j].reshape((-1, 1))))
            # print progress
        loss_ep = np.mean(loss_trace)
        loss_epoch.append(loss_ep)        

        # validation after each epoch
        loss_valid_list = []
        for j in range(test_x_set.shape[1]):
            x = test_x_set[:, j].reshape((-1, 1))
            y = test_y_set[:, j].reshape((-1, 1))
            y_pred = cmac_predictor.predict(x)
            loss_valid = cmac_predictor.compute_loss(y_pred, y)
            loss_valid_list.append(loss_valid)
        loss_valid_ep = np.mean(loss_valid_list)
        loss_valid_epoch.append(loss_valid_ep)
        print("Epoch %d: loss %.8f, validation loss %.8f" % (i, loss_ep, loss_valid_ep))

    # test show result on training
    # Show some prediction on training dataset
    test_training_x_set = training_set_matrix[:, np.arange(1, 151, 10)]
    test_training_y_set = training_label_matrix[:, np.arange(1, 151, 10)]
    for i in range(15):
        print("Test on training %d" % i)
        x = test_training_x_set[:, i].reshape((-1, 1))
        y = test_training_y_set[:, i].reshape((-1, 1))
        print("Coord: [%d, %d]" % (x[0] * 320, x[1] * 240))
        # print(y)
        y_pred = cmac_predictor.predict(x)
        # print(y_pred)
        loss = cmac_predictor.compute_loss(y_pred, y)
        print(loss)

    # Show some prediction on test dataset
    loss_list = []
    for i in range(test_x_set.shape[1]):
        print("Test set %d" % i)
        x = test_x_set[:, i].reshape((-1, 1))
        y = test_y_set[:, i].reshape((-1, 1))
        print("Coord: [%d, %d]" % (x[0] * 320, x[1] * 240))
        # print(y)
        y_pred = cmac_predictor.predict(x)
        # print(y_pred)
        loss = cmac_predictor.compute_loss(y_pred, y)
        print(loss)
        loss_list.append(loss)
    print("Test loss %.8f" % np.mean(np.array(loss_list)))

    # save parameters in track_cmac.txt
    # cmac_predictor.save_parameter("track_cmac")

    plt.figure()
    plt.subplot(211)
    plt.plot(loss_epoch)
    plt.ylim([-0.1, 10])
    plt.title("Training loss, %d data points, receptive_field %d" % (train_x_set.shape[1], receptive_field))
    plt.subplot(212)
    plt.plot(loss_valid_epoch)
    plt.title("Validation loss, %d data points, receptive_field %d" % (train_x_set.shape[1], receptive_field))
    plt.show()

