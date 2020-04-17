from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


def deprocess(x):
    return np.reshape((x * 255.0), (-1, 28, 28)).astype(np.uint8)


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
