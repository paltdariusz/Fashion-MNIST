# Fashion-MNIST 

## Introduction
**Image classification** using **KNN** and **CNN**. Fashion-MNIST is a dataset of Zalando's article images - consisting
of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated 
with a label from 10 classes. More about [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
Our goal is to build a **K-NN model** and a **CNN** to classify an image with cloth to which class it belongs.


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


## Built With

* [Python 3.8](http://docs.python.org/3/) - Programming language
* [numpy](https://www.numpy.org/doc/) - Packge for scientific computing with Python
* [PyTorch](https://pytorch.org/docs/stable/) - Deep learning tensor library


## Authors

* **Dariusz Palt** - *Initial work* - [paltdariusz](https://github.com/paltdariusz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
