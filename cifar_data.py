import numpy as np
import pickle

IMAGE_SIZE = 32

train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_files = ['test_batch']

def preprocess(x):
    return x.astype(np.float) / 255.0

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar(data_path='/home/dchen/dataset/cifar-10-batches-py/'):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in train_files:
        data = unpickle(data_path + file)
        x_train.append(data[b'data'])
        y_train += data[b'labels']

    for file in test_files:
        data = unpickle(data_path + file)
        x_test.append(data[b'data'])
        y_test += data[b'labels']

    x_train = np.concatenate(x_train, axis=0).reshape(-1, 3, 32, 32)
    x_test = np.concatenate(x_test, axis=0).reshape(-1, 3, 32, 32)

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test