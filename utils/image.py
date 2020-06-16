import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    msg = f"{CLASS_NAMES[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({CLASS_NAMES[true_label]})"
    plt.xlabel(msg, color=color)


def plot_value_array(i, predictions_array, true_label=None):
    if not true_label is None:
        predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#969696')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    if not true_label is None:
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('green')
    else:
        thisplot[predicted_label].set_color('blue')


def image_processing(name):
    while True:
        try:
            img = Image.open(name)
            break
        except Exception as e:
            print(f"[{e}] Non such file in directory!")
            name = input("To quit press 'q' or enter valid image name: ")
            if name == 'q':
                exit('INVALID IMAGE NAME')

    if img.size != (28, 28):
        r_img = img.resize((28, 28))
    r_img = ImageOps.invert(r_img)
    r_img = r_img.convert('L')
    r_img = np.array(r_img) / 255
    return r_img


if __name__ == '__main__':
    image_processing('test.jpg')
