from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import os
import cv2
import random
import torch

batch_size = 64

def preprocess(x):
    return x.astype(np.float) / 255.0

def split_two_class(x, y, twoclass):
    id1 = np.where(y == twoclass[0])[0]
    id2 = np.where(y == twoclass[1])[0]
    x_class1 = x[id1, :]
    x_class2 = x[id2, :]
    x = np.concatenate((x_class1, x_class2), axis=0)
    y = np.ones(shape=x.shape[0], dtype=np.int32)
    y[0: id1.shape[0]] = 0

    return x, y

def load_mnist(reshape=True, onehot=True, twoclass=None, binary=False, prep=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if twoclass is not None and len(twoclass) == 2:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
        x_train, y_train = split_two_class(x_train, y_train, twoclass)
        x_test, y_test = split_two_class(x_test, y_test, twoclass)

    if reshape:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)
    else:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)

    if onehot:
        classes = 10 if twoclass is None else 2
        y_train = to_categorical(y_train, num_classes=classes)
        y_test = to_categorical(y_test, num_classes=classes)

    if binary:
        x_temp = np.zeros(x_train.shape, x_train.dtype)
        x_temp[np.where(x_train > 127)] = 255
        x_train = x_temp

        x_temp = np.zeros(x_test.shape, x_test.dtype)
        x_temp[np.where(x_test > 127)] = 255
        x_test = x_temp

    if prep:
        x_train = preprocess(x_train)
        x_test = preprocess(x_test)

    return x_train, y_train, x_test, y_test

def load_data(data_path, binary=False, prep=True):
    data_pairs =[]

    labels = os.listdir(data_path)
    for label in labels:    
        filepath = data_path + label
        filename  = os.listdir(filepath)
        for fname in filename:
            ffpath = filepath + "/" + fname
            data_pair = [ffpath, int(label)]
            data_pairs.append(data_pair)
    
    data_cnt = len(data_pairs)
    data_x = np.empty((data_cnt, 1, 28, 28), dtype="float32")
    data_y = []

    random.shuffle(data_pairs)

    i = 0
    for data_pair in data_pairs:
        img = cv2.imread(data_pair[0], 0)
        img = cv2.resize(img, (28, 28))
        arr = np.asarray(img, dtype="float32")
        data_x[i, :, :, :] = arr
        i += 1
        data_y.append(data_pair[1])
        
    if binary:
        x_temp = np.zeros(data_x.shape, data_x.dtype)
        x_temp[np.where(data_x > 127)] = 255
        data_x = x_temp

    if prep:
        data_x = preprocess(data_x)

    data_y = np.asarray(data_y)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    dataset = torch.utils.data.TensorDataset(data_x, data_y)
            
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    return loader