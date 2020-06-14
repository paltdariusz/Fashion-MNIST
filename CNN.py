import sklearn
import numpy as np
import torch
import torchvision as tv
from utils import mnist_reader
import matplotlib.pyplot as plt
from augmentation import augument
import time


def load_data():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    return X_train, y_train, X_test, y_test


st = time.time()
X_train, y_train, X_test, y_test = load_data()

X_train, y_train = augument(X_train, y_train)
end = time.time()

print(end - st)
print(X_train.shape, y_train.shape)
