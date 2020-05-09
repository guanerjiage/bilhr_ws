#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
import struct


def unpack_mnist_image(filename):
    with open(filename, "rb") as f:
        data = f.read()
    
    magic_num, image_num, row_num, col_num = struct.unpack_from(">4I", data, offset=0) # 4 unsigned int header
    print("Loading %d images from " % image_num + filename)
    offset = 16 
    dataset = np.zeros((image_num, row_num, col_num)) # allocate memory
    fmt = ">%dB" % (row_num * col_num)
    off_increment = struct.calcsize(fmt)
    for i in range(image_num):
        img = struct.unpack_from(fmt, data, offset)
        dataset[i] = np.array(img).reshape((row_num, col_num))
        offset += off_increment

    return dataset

def unpack_mnist_label(filename):
    with open(filename, "rb") as f:
        data = f.read()
    
    magic_num, image_num = struct.unpack_from(">2I", data, offset=0) # 2 unsigned int header
    print("Loading %d labels from " % image_num + filename)
    offset = 8 
    dataset = np.zeros((image_num)) # allocate memory
    fmt = ">1B"
    for i in range(image_num):
        label = struct.unpack_from(fmt, data, offset)
        dataset[i] = np.array(label)
        offset += 1

    return dataset

if __name__ == "__main__":
    training_set = unpack_mnist_image("train-images.idx3-ubyte")
    training_label = unpack_mnist_label("train-labels.idx1-ubyte")
    test_set = unpack_mnist_image("t10k-images.idx3-ubyte")
    test_label = unpack_mnist_label("t10k-labels.idx1-ubyte")
    print(training_set.shape)
    print(training_label.shape)
    print(test_set.shape)
    print(test_label.shape)
    print(np.mean(training_set))
    print(np.std(training_set))
    training_set = (training_set - np.mean(training_set)) / np.std(training_set)
    print(np.mean(training_set))
    print(np.std(training_set))
    plt.figure()
    for i in range(1, 4):
        for j in range(1, 9):
            plt.subplot(3, 8, (i - 1) * 8 + j)
            index = np.random.randint(0, 59999)
            plt.imshow(training_set[index])
            plt.title(training_label[index])
    plt.show()





