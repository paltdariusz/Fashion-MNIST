import numpy as np
import cv2
from skimage import transform


def anticlockwise_rotation(X, Y):
    """
    Odwróć każde ze zdjęć przeciwnie z ruchem wskazówek zegara o dany kąt

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i, :, :] = transform.rotate(X[i, :, :], np.random.randint(0, 180), preserve_range=True)
    return nX, Y


def clockwise_rotation(X, Y):
    """
    Odwróć każde ze zdjęć zgodnie z ruchem wskazówek zegara o dany kąt

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i, :, :] = transform.rotate(X[i, :, :], - np.random.randint(0, 180), preserve_range=True)
    return nX, Y


def h_flip(X, Y):
    """
    Odwróć każde ze zdjęć horyzontalnie (w poziomie)

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = np.fliplr(X[i])
    return nX, Y


def v_flip(X, Y):
    """
    Odwróć każde ze zdjęć wertykalnie (w pionie)

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = np.flipud(X[i])
    return nX, Y


def add_blur(X, Y):
    """
    Rozmaż każde ze zdjęć

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    nX = np.zeros_like(X)
    for i in range(X.shape[0]):
        nX[i] = cv2.GaussianBlur(X[i], (9, 9), 0)
    return nX, Y


def augument(X, Y):
    """
    Zastosuj dla zbioru zdjęć *X* oraz *Y* każde z przekształceń

    :param X: Macierz zdjęć
    :param Y: Etykiety klas
    :return: krotka X,Y, gdzie X to macierz z przekształconymi zdjęciami, Y to wektor etykiet klas
        odpowiadający przekształconym zdjęciom
    """
    X = X.reshape((X.shape[0], 28, 28))
    Y = Y.reshape((Y.shape[0], 1))
    nX = X.copy()
    nY = Y.copy()
    functions = [anticlockwise_rotation, clockwise_rotation, h_flip, v_flip, add_blur]
    for function in functions:
        tempX, tempY = function(X.copy(), Y.copy())
        nX = np.vstack((nX, tempX))
        nY = np.vstack((nY, tempY))
    return nX, nY
