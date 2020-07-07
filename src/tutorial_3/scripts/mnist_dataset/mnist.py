#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import urllib2
import subprocess


def download_mnist_dataset(path):    
    url_web = "http://yann.lecun.com/exdb/mnist/"
    zip_names = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    file_names = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"]
    
    for i in range(4):
        file_name = file_names[i]
        zip_name = zip_names[i]
        zip_path = os.path.join(path, zip_name)
        # check if dataset already exists, if yes, do nothing
        if check_exist(path, file_name):
            print(file_name + " already exists")
        # check if zip file already downloaded, if yes, unzip
        elif check_exist(path, zip_name):
            print(zip_name + " already exists")            
            unzip_file(zip_path)
        # else download and unzip
        else:
            download_file(url_web + zip_names[i], path)
            unzip_file(zip_path)

# check if file already exist
def check_exist(path, filename):
    filepath = os.path.join(path, filename)
    return os.path.exists(filepath)

def download_file(url, path):
    # get file name from url
    filename = url.split("/")[-1]
    filepath = os.path.join(path, filename)
    # download
    print("Downloading " + url + " to " + filepath)
    f = urllib2.urlopen(url).read()    
    with open(filepath, "wb") as zip:
        zip.write(f)

def unzip_file(filepath):
    cmd = ["gzip", "-dN", filepath]
    print("Unzip " + filepath)
    subprocess.call(cmd)
    

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
    download_mnist_dataset("./")
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





