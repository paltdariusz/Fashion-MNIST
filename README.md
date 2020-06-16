# Fashion-MNIST 

## Introduction
**Image classification** using **KNN** and **CNN**. Fashion-MNIST is a dataset of Zalando's article images - consisting
of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated 
with a label from 10 classes. More about [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
Our goal is to build a **K-NN model from scratch** and a **CNN using tensorflow library** to classify an image of cloth to which class it belongs.


## Methods

### K-NN

#### Why KNN?
K-Nearest Neighbors is one of the most popular non-parametric models, because of its implementation simplicity and it is vary easy to explain it to newbie.

[K-NN](docs/knn1.png?raw=True "KNN")

So the idea is quite simple. Our model's predicition is based on training set and also we don't need feature extraction 
nor learning algorithms to make it work! First thing we have to define is which distance metrics we are going to use. 
In our model I've implemented 3 different distance metrics (i.e. Hamming, Modified Hamming and Euclidean). Why? To get a 
clear view how different metrics affect on results. Next is to calculate distance from our new point to existing points.
So for example we have in training set 1000 points and one new point (which we want to classify), so we need compute 1000
distances. Next we want to sort ascending our labels in view of the distance. Now it is time to use our hiperparameter `k-value`.
Depending on it we are constricting sorted labels, e.g. we have `len(labels) = 1000` and `k = 5` so our new array is `labels = labels[:5]`.
Our predicted label is one, which occurs the most frequently. That's it! For more info click [here](https://www.youtube.com/watch?v=HVXime0nQeI).

### CNN
**1. Why CNN?**





Model summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 3, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 576)               0
_________________________________________________________________
dense (Dense)                (None, 64)                36928
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

## Results
### KNN
```
Training time: 155 seconds
Accuracy: 73.55% => 7,355 of 10,000
Error: 26.45% => 2,645 of 10,000
Best k: 41 [tested range: <1,199>]
Distance: Modified Hamming
```
```
Training time: 4,186 seconds
Accuracy: 85.27% => 8,527 of 10,000
Error: 14.73% => 1,473 of 10,000
Best k: 3 [tested range: <1,199>]
Distance: Euclidean
```
```
This distance function was tested on smaller set containing only *100* images, 
because results were below expectations.

Training time: 41 seconds (On full set it would probablly take about 4100 seconds)
Accuracy: 59% =>  59 of 100
Error: 41% =>  41 of 100
Best k: 3 [tested range: <1,199>]
Distance: Hamming
```

As we can see, the best result we managed to get is by using **Euclidean distance**. We recived accuracy on level **85.27%**, 
but the training time took more than an hour, comparing it to **Modified Hamming distance** which took only 155 seconds 
and accuracy was just 10 percent points lower. 

Comparing our results to the [benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) we would be on the
6th place. The first place is 0.73 percent point ahead in accuracy. 


### CNN

```
313/313 [==============================] - 1s 3ms/step - loss: 0.2780 - accuracy: 0.9000


Model accuracy: 0.8999999761581421 (on test set)
```
As we can see, we are achieving 90% accuracy on test set. Comparing this result to [benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/),
we would get 1st place (second place is 0.002 percent points behind **big victory**!). Training time on device without 
CUDA support is about 40-50 minutes (with CUDA would be much faster).  

![Model predictions from test set](docs/predictions.png?raw=true "Model predictions from test set")

On image above we can observe how our model classified ten random images and how high was the probability to each class.
 
![Model predictions from given image](docs/predicted_from_jpg.png?raw=true "Model predictions from given image")

On image above we can observe how our model classified 'test.jpg' and how high was the probability to each class.

| Model type  | Accuracy | Training Time |
| :--- | :---: | ---: |
| KNN  | 85.27% | 01:09:45 |
| KNN (bench)  | 86.00% | 00:41:53 |
| CNN | 90.00% | 00:45:13 |
| Best benchmark| 89.70% | 01:12:39 |

## Usage
1. Download project typing to terminal/cmd `git clone https://github.com/paltdariusz/Fashion-MNIST.git`
2. Go to the downloaded directory `cd Fashion-MNIST`
3. Download missing obligatory [extensions](##-Built-With) 
4. Run program by typing into cmd/terminal `python main.py`
5. Select which models you want to train (KNN, CNN, BOTH)<br />
   **TRAINING KNN TAKES ABOUT 1 HOUR! IF YOU WANT TO SEE CNN TRAINING DELETE [THIS](utils/cnn_model.h5) FILE**
   **TO RUN FULL KNN YOU NEED TO CHANGE IN [MAIN](main.py) LINE 6 VARIABLE<br />
   `SKIP_HAMMING = True` TO `SKIP_HAMMING = False`**
6. After training CNN (if chosen) you will be ask to enter filename of your **own** picture to classify (if you dont have I've put 'test.jpg' to directory).<br />
   **If you want to check your own image, put it into this directory and when program asks you write filename, write it as 'filename.extension'**
7. You can find trained model [here](utils/cnn_model.h5), if you want to use it you'll have to do: 
   ```
   from tensorflow import keras
   model = model = keras.models.load_model('utils/cnn_model.h5')
   ```
   **After this you will be able to use trained model functionality!**

Training and test data are in: [data/fashion/](data/fashion) and will be **automatically downloaded if you follow instruction!**

## Built With

* [Python 3.8](http://docs.python.org/3/) - Programming language
* [numpy](https://www.numpy.org/doc/) - Packge for scientific computing with Python
* [TensorFlow](https://tensorflow.org/api_docs/) - Deep learning tensor library
* [PIL](https://pillow.readthedocs.io/) - Python Imaging Library
* [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) - Computer vision algorithms library
* [scikit-image](https://scikit-image.org/docs/dev/index.html) - Image processing library
* [matplotlib](https://matplotlib.org/3.2.1/contents.html) - Vizualization with Python

## Authors

* **Dariusz Palt** - *Initial work* - [paltdariusz](https://github.com/paltdariusz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
