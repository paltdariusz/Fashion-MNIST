import numpy as np
import time

from utils import mnist_reader
import content as ct

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

k = range(1, 201, 2)

X_train_n, X_test_n = X_train / 255, X_test / 255
start = time.time()
error, best_k, errors = ct.model_selection_knn(X_test_n, X_train_n, y_test, y_train, k)
end = time.time()

print(error, best_k, errors)
print(f"Uczenie zajęło: {end - start}")
