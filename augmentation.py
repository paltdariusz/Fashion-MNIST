import numpy as np
import cv2
import skimage
from skimage import transform


def anticlockwise_rotation(X, Y):
    """
    Odwróć każde ze zdjęć przeciwnie z ruchem wskazówek zegara o dany kąt

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = transform.rotate(X[i], np.random.randint(0, 180))
    X = np.vstack((X, nX))
    Y = np.vstack((Y.copy(), Y.copy()))
    return X, Y


def clockwise_rotation(X, Y):
    """
    Odwróć każde ze zdjęć zgodnie z ruchem wskazówek zegara o dany kąt

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = transform.rotate(X[i], - np.random.randint(0, 180))
    X = np.vstack((X, nX))
    Y = np.vstack((Y, Y))
    return X, Y


def h_flip(X, Y):
    """
    Odwróć każde ze zdjęć horyzontalnie (w poziomie)

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = np.fliplr(X[i])
    X = np.vstack((X, nX))
    Y = np.vstack((Y, Y))
    return X, Y


def v_flip(X, Y):
    """
    Odwróć każde ze zdjęć wertykalnie (w pionie)

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = np.flipud(X[i])
    X = np.vstack((X, nX))
    Y = np.vstack((Y, Y))
    return X, Y


def add_noise(X, Y):
    """
    Dodaj szumu do każdego zdjecia

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = skimage.util.random_noise(X[i])
    X = np.vstack((X, nX))
    Y = np.vstack((Y, Y))
    return X, Y


def add_blur(X, Y):
    """
    Rozmaż każde ze zdjęć

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = cv2.GaussianBlur(X[i], (9, 9), 0)
    X = np.vstack((X, nX))
    Y = np.vstack((Y, Y))
    return X, Y


def augument(X, Y):
    """
    Zastosuj dla zbioru zdjęć *X* oraz *Y* każde z przekształceń

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to powiększona macierz ze zdjęciami o rotacje, Y to powiększony wektor etykiet klas
        odpowiadający stworzonym zdjęciom
    """
    X = X.reshape((X.shape[0], 28, 28))
    Y = Y.reshape((Y.shape[0], 1))
    functions = [anticlockwise_rotation, clockwise_rotation, h_flip, v_flip, add_noise, add_blur]
    for function in functions:
        X, Y = function(X, Y)
    X = X.reshape((X.shape[0], 28 * 28))
    Y = Y.flatten()
    return X, Y
