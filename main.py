import matplotlib.pyplot as plt
import numpy as np
from utils import mnist_reader, augmentation as aug, CNNetwork as cnn, image as img, KNN

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    return X_train, y_train, X_test, y_test


def normalize(X_train, X_test):
    return X_train / 255, X_test / 255


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    msg = "Press:\n1. start only KNN,\n2. start only CNN,\n3. start both\n"
    while True:
        try:
            choice = int(input(msg))
            if 4 > choice > 0:
                break
            else:
                print("Wrong number!\n")
        except Exception as e:
            print(f"[{e}] Not a number!\n")

    # ------------KNN PART------------------------
    if choice != 2:
        print("\n-----!STARTING KNN IT WILL TAKE ~ 1HR!-----")
        k = range(1, 201, 2)
        X_train_n, X_test_n = normalize(X_train.copy(), X_test.copy())
        hamming, hamming_m, euklidean = KNN.distance_selection_knn(X_test_n, X_train_n, y_test, y_train, k)
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
    if choice != 1:
        # rozszerzamy zbiór o przekształcone zdjęcia
        X_train_aug, y_train_aug = aug.augument(X_train.copy(), y_train.copy())
        # normalizujemy wartości zdjęć - do przedziału (0,1]
        X_train_aug, X_test_n = normalize(X_train_aug, X_test.copy())
        # dopasowujemy dane, tak aby model je przyjął
        X_train_aug = X_train_aug.reshape((X_train_aug.shape[0], 28, 28, 1))
        X_test_n = X_test_n.reshape((X_test_n.shape[0], 28, 28, 1))
        # trenujemy model lub wczytujemy jeżeli istnieje w ścieżce
        model = cnn.train(X_train_aug, y_train_aug)
        # drukujemy podstawowe dane modelu
        print(model.summary())
        # testujemy wyuczony model na danych testowych
        test_loss, test_acc = model.evaluate(X_test_n, y_test)
        print(f"\n\nModel accuracy: {test_acc}")
        predictions = cnn.predict(model, X_test_n)
        # plot random 10 test images, their labels, and true labels
        plt.figure(figsize=(8, 10))
        idx = np.random.choice(X_test_n.shape[0], 10)
        for i in range(len(idx)):
            plt.subplot(5, 4, 2 * i + 1)
            img.plot_image(idx[i], predictions[idx[i]], y_test, X_test_n)
            plt.subplot(5, 4, 2 * i + 2)
            img.plot_value_array(idx[i], predictions[idx[i]], y_test)
        plt.tight_layout()
        plt.show()
        img_name = input("Enter image name [name.jpg](if you don't have any enter: 'test.jpg'): ")

        org_img, new_image = img.image_processing(img_name)
        new_image = new_image.reshape((1, 28, 28, 1))
        predictions_single = cnn.predict(model, new_image)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        img.plot_image(1, predictions_single[0], None, org_img)

        plt.subplot(1, 2, 2)
        img.plot_value_array(1, predictions_single[0])
        _ = plt.xticks(range(10), CLASS_NAMES, rotation=45)
        plt.show()
