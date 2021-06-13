# CIFAR-10 Image Classification using CNN Model and Pretrained AutoEncoder CNN Model 
Task is to design a network that combines supervised and unsupervised architectures in one model to achieve a classification task.

## The CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each: 
![cifar10 image](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/cifat10.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks. 

## Problem Statement
The model must start with autoencoder(s) (stacking autoencoder is ok) that is connected through its hidden layer to another network of choice, as shown in the figure below:
![problem architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/Problem_architecture.png)
This autoencoder takes an input (image) at node 1 and reconstructs it at its output at node 3. It creates valuable features at its hidden layers (node 2) during this process. it is hypothesized that if node 2 is used as input for the CNN (node 4) then the classification can be improved.
#### Data:
We want to use this model to classify Cifar-10 dataset by using at max 2500 training images for each of the **bird**, **deer**, and **truck** classes while using 5000 for the other classes. The model should be evaluated by the test set of 10000 images (1000 for each class).

## 





## CNN Architecture

![CNN Architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_architecture.png)






