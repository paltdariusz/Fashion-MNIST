# Fashion-MNIST 

## Introduction
**Image classification** using **KNN** and **CNN**. Fashion-MNIST is a dataset of Zalando's article images - consisting
of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated 
with a label from 10 classes. More about [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
Our goal is to build a **K-NN model** and a **CNN** to classify an image of cloth to which class it belongs.


## Methods

### K-NN

### CNN

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
This distance function was tested on smaller set containing only *100* images, because results were below expectations.

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
Model summary

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

```
313/313 [==============================] - 1s 3ms/step - loss: 0.2780 - accuracy: 0.9000


Model accuracy: 0.8999999761581421 (on test set)
```


![Model predictions from test set](docs/predictions.png?raw=true "Model predictions from test set")



![Model predictions from given image](docs/predicted_from_jpg.png?raw=true "Model predictions from given image")


## Built With

* [Python 3.8](http://docs.python.org/3/) - Programming language
* [numpy](https://www.numpy.org/doc/) - Packge for scientific computing with Python
* [TensorFlow](https://tensorflow.org/api_docs/) - Deep learning tensor library
* [PIL](https://pillow.readthedocs.io/)
* [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
* [scikit-image](https://scikit-image.org/docs/dev/index.html)
* [matplotlib](https://matplotlib.org/3.2.1/contents.html)

## Authors

* **Dariusz Palt** - *Initial work* - [paltdariusz](https://github.com/paltdariusz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
