import numpy as np
import time as t


def hamming_distance_m(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    M = (np.ones_like(X) - X) @ X_train.T + X @ (np.ones_like(X_train) - X_train).T

    return M


def euclidean_distance(X, X_train):
    """
    Zwróć odległość Euklidesa dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    M = np.zeros((X.shape[0], X_train.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i][j] = np.linalg.norm(X[i] - X_train[j])

    return M


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    M = np.zeros((X.shape[0], X_train.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i][j] = np.count_nonzero(X[i] - X_train[j])

    return M


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    y_sorted = y[np.argsort(Dist, kind='mergesort')]

    return y_sorted


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    neighbors = y[:, :k]
    probability = np.apply_along_axis(np.bincount, 1, neighbors, minlength=np.max(y) + 1)

    return np.true_divide(probability, k)


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    max_p = p_y_x.shape[1] - 1 - np.argmax(np.flip(p_y_x, 1), axis=1)
    error = np.count_nonzero(max_p - y_true)

    return np.true_divide(error, y_true.shape[0])


def model_selection_knn(X_val, X_train, y_val, y_train, k_values, f_distance):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :param f_distance: funkcja do liczenia odległości
    :return: krotka (best_error, best_k, errors, time), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values" i time - czas trwania
    """
    start = t.time()
    errors = np.zeros_like(k_values, dtype='float64')
    sorted_labels = sort_train_labels_knn(f_distance(X_val, X_train), y_train).astype('uint8')
    for i in range(len(errors)):
        prob = p_y_x_knn(sorted_labels, k_values[i])
        errors[i] = classification_error(prob, y_val)
    end = t.time()

    return np.min(errors), k_values[np.argmin(errors)], errors, end - start


def distance_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych *odległości*. Dokonaj selekcji modelu KNN
    wybierając optymalną metodę liczenia *odległości*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotkę tablic składających się z (best_error, best_k, errors, time),
        gdzie "best_error" to najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values" i time - czas trwania
    """
    distance_functions = [hamming_distance, hamming_distance_m, euclidean_distance]
    results = []
    for fun in distance_functions:
        results.append(model_selection_knn(X_val.copy(), X_train.copy(), y_val.copy(), y_train.copy(), k_values, fun))

    return results[0], results[1], results[2]
