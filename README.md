# CIFAR-10 Image Classification using CNN Model and Pretrained AutoEncoder CNN Model 
Task is to design a network that combines supervised and unsupervised architectures in one model to achieve a classification task.

## The CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each: 
![cifar10 image](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/cifar10.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks. 

## Problem Statement
The model must start with autoencoder(s) (stacking autoencoder is ok) that is connected through its hidden layer to another network of choice, as shown in the figure below:
![problem architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/Problem_architecture.png)
This autoencoder takes an input (image) at node 1 and reconstructs it at its output at node 3. It creates valuable features at its hidden layers (node 2) during this process. it is hypothesized that if node 2 is used as input for the CNN (node 4) then the classification can be improved.

We want to use this model to classify Cifar-10 dataset by using at max 2500 training images for each of the **bird**, **deer**, and **truck** classes while using 5000 for the other classes. The model should be evaluated by the test set of 10000 images (1000 for each class).

## Data Preprocessing
For bird, deer and truck class out of 5000 data randomly selected 2500 data for each class as train data. Train data is splited into Train and Validation set with split size = 0.15. Here I am using the test data of 10000 images (1000 for each class). Data distribution of train and validation set after making the imbalanced dataset as follows:
![Data Distribution](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/data_distribution.png)

Data is normalized before feeding to training or testing.
I used data augmentation to overcome overfitting. Data augmentation parameters are as follows:
* rotation_range=40
* width_shift_range=0.2
* height_shift_range=0.2
* shear_range=0.2
* zoom_range=0.2
* horizontal_flip=True
* fill_mode='nearest'

Though I used data augmentation for the imbalanced data total number of data is half than the other classes. To overcome this I used balanced data generator function which will randomly oversample the imbalanced data.





## CNN Architecture

![CNN Architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_architecture.png)



In the CNN architecture BatchNormalization is used after each Convolutional layer and after first Dense Layer Dropout with probability 0.4 is used as regularizer. After BatchNormalization layer Relu Aactivation is used.
### Parameters
* batch_size = 128
* epochs = 100
* learning rate = 0.001
* loss function = categorical_crossentropy
* optimizer = Adam

To check overfitting below callback parameters used,
* Early stopping with patience = 15 and monitor = 'val_loss'. If validation loss didn't improve within 15 epochs the model stop training.
* ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
** parameters used: patience=10,monitor="val_loss",factor=0.3, min_lr=0.0001, verbose=1,cooldown=1
** This callback monitors "val_loss" and if no improvement is seen for 10 epochs, the learning rate is reduced by factor 0.3

#### Stratified ShuffleSplit cross-validator
This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. Here 5 fold cross validation is used.
Confusion matrix for one of the cross validation are as follows:
```
          precision    recall  f1-score   support

    airplane       0.78      0.74      0.76      1000
  automobile       0.70      0.94      0.80      1000
        bird       0.79      0.51      0.62      1000
         cat       0.66      0.49      0.56      1000
        deer       0.76      0.65      0.70      1000
         dog       0.67      0.71      0.69      1000
        frog       0.66      0.88      0.75      1000
       horse       0.72      0.83      0.77      1000
        ship       0.92      0.81      0.86      1000
       truck       0.77      0.80      0.78      1000

    accuracy                           0.73     10000
   macro avg       0.74      0.73      0.73     10000
weighted avg       0.74      0.73      0.73     10000
```
To see the other cross validation confusion matrix see this file, 
[Confusion matrix CNN model](https://github.com/shakhaout/cifar10_classification/blob/main/checkpoints/CNN_classification_report.txt)

# Train
To train the CNN model run below command:
```
python main.py --train TRAIN --cnn CNNN
```
To train the AutoEncoder CNN model run below command:
```
python main.py --train TRAIN --autoencoder AUTOENCODER
```
For the first time this command will train the autoencoder model. After finishing the autoencoder model, using the weights of this autoencoder model CNN model will start training. If pretrained autoencoder weight already exists in the checkpoints directory, running above command will train the CNN model and inputs of this CNN model will be the pretrained weights.


