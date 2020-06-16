import tensorflow as tf
from tensorflow import keras
import os


def train(X_train, y_train):
    if os.path.exists('utils/cnn_model.h5'):
        model = keras.models.load_model('utils/cnn_model.h5')
    else:
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(10))

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=64)
        model.save('utils/cnn_model.h5')

    return model


def predict(model, X_test):
    probability_model = keras.Sequential()
    probability_model.add(model)
    probability_model.add(keras.layers.Softmax())
    predictions = probability_model.predict(X_test)
    return predictions
