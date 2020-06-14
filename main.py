from utils import mnist_reader
import content as ct


def load_data():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    return X_train, y_train, X_test, y_test


def normalize(X_train, X_test):
    return X_train / 255, X_test / 255


if __name__ == '__main__':
    # ------------KNN PART------------------------
    k = range(1, 201, 2)
    X_train, y_train, X_test, y_test = load_data()
    X_train_n, X_test_n = normalize(X_train.copy(), X_test.copy())
    hamming, hamming_m, euklidean = ct.distance_selection_knn(X_test_n[:100], X_train_n, y_test[:100], y_train, k)
    distances = ['Hamming distance', 'Modified Hamming distance', 'Euclidean distance']
    results = [hamming, hamming_m, euklidean]
    best = min(hamming[0], hamming_m[0], euklidean[0])
    for i in range(len(results)):
        if best == results[i][0]:
            print("!!BEST ACCURACY!!")
        print(f"-------{distances[i]}-------")
        print(f"Accuracy: {1 - results[i][0]}")
        print(f"Error: {results[i][0]}")
        print(f"Best k: {results[i][1]}")
        print(f"Training time: {results[i][-1]:.2f}")
        print((14 + len(distances[i])) * '-')
        print()

    # ------------CNN PART------------------------
