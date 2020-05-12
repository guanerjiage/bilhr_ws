#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np

if __name__ == "__main__":
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
    # pitch [-2, 2] to [0, 1], roll [0, 1] no normalization need
    training_label_matrix[0, :] = (training_label_matrix[0, :] + 2) / 4.0

    # save txt
    np.savetxt("training_x_norm.txt", training_set_matrix)
    np.savetxt("training_y_norm.txt", training_label_matrix)