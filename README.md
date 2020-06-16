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
#### Why CNN?
Convolutional Neural Network is the most popular and very good neural network model at image classification. As we can see in **Model summary** table
the number of parameters grwos rapidly with the increase in the number of layers.  Training model using diferent neural 
networks than CNN can take much longer. The time taken for tuning these parameters is decreased thanks to CNN. In features
extraction CNN is trained to identify the edges of objects in any image! Summing up, I've chosen it because it is very popular and 
thanks to TenserFlow library easy to implement!

#### Training process

##### Data augumentation
I've managed to get some (**300k**) extra training images by changing them (eg. horizontal/vertical rotation, rotatate 
(anti)clockwise with random angle, add blur). So now out training set contains **360k** images! That should improve model precision.

Next step is normalizing the data. Neural networks love handling data between `[0, 1]` (because when numbers are not normalized,
they can get very big and could cause overflow of datatype), so we must devide X_train and X_test by 255 (maximum pixel value).

As I said CNN is automatically extracts features so we didn't have to deal with it. 

##### Creating CNN model
I am using very common pattern for a CNN model: stack of **Conv2D** and **MaxPooling2D** layers, beteen them I added dropout 
layer to minimalize chance of exsiting negative effect called overfitting (when model predict much accurate train set than test set).
To prevent overfitting I splitted training set to new training set (80%) and vilidation set (20%). 

What is Conv2D?

[Conv2D](docs/conv.gif?raw=True "Conv2D")
[Conv2D](docs/conv1.gif?raw=True "Conv2D")

What is MaxPooling2D? 
Reducing a matrix (or matrices) created by an earlier convolutional layer to a smaller matrix.
Pooling usually involves taking either the maximum or average value across the pooled area.
[Pooling](docs/Pooling.svg?raw=True "Conv2D")

What is Dense layer?
It computes `output = activation(dot(input, kernel) + bias)`

Why 'adam' optimizer? 
It is the fastest and gives us best results.

Why Sparse Categorical Crossentropy as loss function?
Again it gives us the best results.

Why ReLU? It is the most commonly used activation function in neural networks (especially in CNNs). It is cheap to compute as there is no complicated math, 
therefore training might take less time! 

##### Training!
```
Epoch 1/10
4500/4500 [==============================] - 154s 34ms/step - loss: 0.7137 - accuracy: 0.7378 - val_loss: 1.6641 - val_accuracy: 0.4543
Epoch 2/10
4500/4500 [==============================] - 133s 30ms/step - loss: 0.5028 - accuracy: 0.8163 - val_loss: 2.1343 - val_accuracy: 0.3995
Epoch 3/10
4500/4500 [==============================] - 113s 25ms/step - loss: 0.4502 - accuracy: 0.8357 - val_loss: 2.0292 - val_accuracy: 0.4084
Epoch 4/10
4500/4500 [==============================] - 112s 25ms/step - loss: 0.4210 - accuracy: 0.8460 - val_loss: 2.8115 - val_accuracy: 0.3807
Epoch 5/10
4500/4500 [==============================] - 111s 25ms/step - loss: 0.4033 - accuracy: 0.8524 - val_loss: 2.4198 - val_accuracy: 0.3839
Epoch 6/10
4500/4500 [==============================] - 129s 29ms/step - loss: 0.3901 - accuracy: 0.8564 - val_loss: 2.9974 - val_accuracy: 0.3678
Epoch 7/10
4500/4500 [==============================] - 115s 26ms/step - loss: 0.3768 - accuracy: 0.8621 - val_loss: 3.7451 - val_accuracy: 0.3493
Epoch 8/10
4500/4500 [==============================] - 140s 31ms/step - loss: 0.3682 - accuracy: 0.8641 - val_loss: 2.9253 - val_accuracy: 0.3778
Epoch 9/10
4500/4500 [==============================] - 131s 29ms/step - loss: 0.3632 - accuracy: 0.8658 - val_loss: 3.4720 - val_accuracy: 0.3462
Epoch 10/10
4500/4500 [==============================] - 134s 30ms/step - loss: 0.3548 - accuracy: 0.8687 - val_loss: 3.4807 - val_accuracy: 0.3600
```

#### Uploading your own image to classification
You can test trained model on your own picture. Fistly program is going to turn it into negative, then resize it to size: 28x28 pxls 
and finally will try to predict class of image. **Pretty cool feature** 


#### Model summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 28)        280
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 28)        0
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 28)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        16192
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
Total params: 90,978
Trainable params: 90,978
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
313/313 [==============================] - 2s 5ms/step - loss: 0.2638 - accuracy: 0.9021


Model accuracy: 0.9021000266075134
```
As we can see, we are achieving 90% accuracy on test set. Comparing this result to [benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/),
we would get 1st place (second place is 0.02 percent points behind **big victory**!). Training time on device without 
CUDA support is about 25:40 minutes (with CUDA would be much faster).  

![Model predictions from test set](docs/predictions.png?raw=true "Model predictions from test set")

On image above we can observe how our model classified ten random images and how high was the probability to each class.
 
![Model predictions from given image](docs/predicted_from_jpg.png?raw=true "Model predictions from given image")

On image above we can observe how our model classified 'test.jpg' and how high was the probability to each class.

| Model type  | Accuracy | Training Time |
| :--- | :---: | ---: |
| KNN  | 85.27% | 01:09:45 |
| KNN (bench)  | 86.00% | 00:41:53 |
| **CNN** | **90.21%** | **00:25:40** |
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

## Sources

* [KNN](https://www.youtube.com/watch?v=HVXime0nQeI)
* [CNN](https://analyticsindiamag.com/convolutional-neural-network-image-classification-overview/)
* [CNN](https://developers.google.com/machine-learning/glossary/#convolutional_neural_network)
* [CNN](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* [Course](https://www.ii.pwr.edu.pl/~zieba/index.php?p=kursy)