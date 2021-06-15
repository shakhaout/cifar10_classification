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

Though I used data augmentation for the imbalanced data total number of data is half than the other classes. To overcome this I used balanced data generator function which will randomly oversample the imbalanced data to keep the same ratio for each class in the mini batch. In CNN classification and AutoEncoder CNN classification used this.



## Training Parameters
Below are common parameters used for both CNN and Pretrained Autoencoder CNN model,
* batch_size = 128
* epochs = 100
* learning rate = 0.001
* loss function = categorical_crossentropy
* optimizer = Adam

To check overfitting below callback parameters used,
* Early stopping with patience = 15 and monitor = 'val_loss'. If validation loss didn't improve within 15 epochs the model stop training.
* ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
 * parameters used: patience=10,monitor="val_loss",factor=0.3, min_lr=0.0001, verbose=1,cooldown=1
 * This callback monitors "val_loss" and if no improvement is seen for 10 epochs, the learning rate is reduced by factor 0.3

#### Stratified ShuffleSplit cross-validator
This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. Here 5 fold cross validation is used.


## 1. CNN Classification
### Architecture

![CNN Architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_architecture.png)



In the CNN architecture BatchNormalization is used after each Convolutional layer and after first Dense Layer Dropout with probability 0.4 is used as regularizer. After BatchNormalization layer Relu Aactivation is used.


### Learning Curve
For first iteration of cross validation accuracy & loss curve are as follows,
![accuracy curve CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_CNN_classification_kfold1.png)
![loss curve CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_CNN_classification_kfold1.png)

To see accuracy plots for other cross validations,
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_CNN_classification_kfold2.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_CNN_classification_kfold3.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_CNN_classification_kfold4.png)
[kfold5](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_CNN_classification_kfold5.png)


To see loss plots for other cross validations,
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_CNN_classification_kfold2.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_CNN_classification_kfold3.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_CNN_classification_kfold4.png)
[kfold5](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_CNN_classification_kfold5.png)

Confusion matrix heatmap for first iteration(recall plot),
![Heatmap of CNN classification](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_kfold1_heatmap.png)

To see other heatmaps,
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_kfold2_heatmap.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_kfold3_heatmap.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_kfold4_heatmap.png)
[kfold5](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/CNN_classification_model_kfold5_heatmap.png)

Classification report for first iteration of the cross validation is as follows:
```
Kfold Iteration:1`
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
[Classification report CNN model](https://github.com/shakhaout/cifar10_classification/blob/main/checkpoints/CNN_classification_report.txt)

## 2. AutoEncoder (with data augmentation)
### Architecture
For Encoder Decoder model I have used modified VGG16 Architecture with some BatchNormalization and Dropout layer as regularizer to check overfitting.

![AutoEncoder Architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/autoencoder_model_architecture.png)

 
#### Parameters
* Loss function = 'mean_squared_error'
* Optimizer = RMSprop (with learning rate 0.001)
* Augmentated data is fitted in the model.

### Learning curve
![Autoencoder Accuracy plot](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuray_plot_autoencoder.png)
![Autoencoder loss plot](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_autoencoder.png)

Peak signal to noise ratio (PSNR) and structural index similarity (SSIM) of the test set and the reconstructed images are as follows on Test data of 10000 examples:

**Average PSNR:17.416**
**Average SSIM: 0.526**

### Reconstructed Image
In test data set some of the reconstructed images of the decoder model are as follows:
![Decoded image](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/autoencoder_org_reconstd_imgs.png)

## 3. AutoEncoder Pretrained CNN Classification
### Architecture

![AutoEncoder CNN Architecture](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_architecture.png)



In this AutoEncoder Pretrained CNN model, after the encoder part same CNN architecture is used as the normal CNN classification. But as inputs autoencoder pretrained layers were used. Here I took pretrained weights upto 11 layers and  kept layer.trainable = False for the first 11 layers.
```
encode = encoder(self.input_image)
clf = Model(self.input_image,classifier(encode))
for l1, l2 in zip(clf.layers[0:11], cnn_autoencoder.layers[0:11]):
   l1.set_weights(l2.get_weights())
for layer in clf.layers[0:11]:
   layer.trainable = False
```

### Learning Curve
For 5th iteration of cross validation accuracy & loss curve are as follows,
![accuracy curve AutoEncoder CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold5.png)
![loss curve AutoEncoder CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold5.png)

To see accuracy plots for other cross validations,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold1.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold2.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold3.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold4.png)


To see loss plots for other cross validations,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold1.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold2.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold3.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold4.png)

Confusion matrix heatmap for 5th iteration(recall plot),
![Heatmap of AutoEncoder CNN classification](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold5_heatmap.png)

To see other heatmaps,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold1_heatmap.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold2_heatmap.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold3_heatmap.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold4_heatmap.png)

Classification report for 5th iteration of the cross validation is as follows:
```
  Kfold Iteration:5
              precision    recall  f1-score   support

    airplane       0.84      0.79      0.81      1000
  automobile       0.87      0.89      0.87      1000
        bird       0.81      0.56      0.67      1000
         cat       0.72      0.53      0.61      1000
        deer       0.70      0.73      0.72      1000
         dog       0.82      0.57      0.67      1000
        frog       0.56      0.95      0.71      1000
       horse       0.83      0.82      0.83      1000
        ship       0.85      0.88      0.87      1000
       truck       0.78      0.89      0.83      1000

    accuracy                           0.76     10000
   macro avg       0.78      0.76      0.76     10000
weighted avg       0.78      0.76      0.76     10000
```
To see the other cross validation confusion matrix see this file, 
[Classification report AutoEncoder CNN model](https://github.com/shakhaout/cifar10_classification/blob/main/checkpoints/AutoEncoder_classification_report.txt)


## 4. AutoEncoder ( with no data augmentation)
Here same architecture is used as in AutoEncoder(No.2) with data augmentation. Other parameters are also similar to previous autoencoder model just no augmenation technique is adopted while trining the model.

### Learning curve
![Autoencoder Accuracy plot no augmentation](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuray_plot_autoencoder_no_augmentation.png)
![Autoencoder loss plot no augmentation](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_autoencoder_no_augmentation.png)

Peak signal to noise ratio (PSNR) and structural index similarity (SSIM) of the test set and the reconstructed images are as follows on Test data of 10000 examples:

**Average PSNR:27.825**  
**Average SSIM: 0.944**

### Reconstructed Image
In test data set some of the reconstructed images of the decoder model are as follows:
![Decoded image no augmentation](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/autoencoder_org_reconstd_imgs_no_aug.png)

## 5. AutoEncoder Pretrained CNN Classification ( Pretrained AutoEncoder of No.4)
Here model architecture and parameters are similar to  previous AutoEncoder CNN classification model(No.3). But AutoEncoder model is trained with no data augmentation and this pretrained weight is used in this model.

### Learning Curve
For 5th iteration of cross validation accuracy & loss curve are as follows,
![accuracy curve AutoEncoder CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold5_noaug.png)
![loss curve AutoEncoder CNN](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold5_noaug.png)

To see accuracy plots for other cross validations,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold1_noaug.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold2_noaug.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold3_noaug.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/accuracy_plot_AutoEncoder_classification_kfold4_noaug.png)


To see loss plots for other cross validations,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold1_noaug.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold2_noaug.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold3_noaug.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/loss_plot_AutoEncoder_classification_kfold4noaug.png)

Confusion matrix heatmap for 5th iteration(recall plot),
![Heatmap of AutoEncoder CNN classification](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold5_heatmap_noaug.png)

To see other heatmaps,
[kfold1](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold1_heatmap_noaug.png)
[kfold2](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold2_heatmap_noaug.png)
[kfold3](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold3_heatmap_noaug.png)
[kfold4](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/AutoEncoder_classification_model_kfold5_heatmap_noaug.png)

Classification report for 5th iteration of the cross validation is as follows:
```
  Kfold Iteration:5
              precision    recall  f1-score   support

    airplane       0.86      0.86      0.86      1000
  automobile       0.89      0.92      0.90      1000
        bird       0.86      0.68      0.76      1000
         cat       0.70      0.67      0.69      1000
        deer       0.82      0.76      0.79      1000
         dog       0.79      0.76      0.77      1000
        frog       0.74      0.93      0.83      1000
       horse       0.86      0.89      0.87      1000
        ship       0.91      0.89      0.90      1000
       truck       0.84      0.89      0.87      1000

    accuracy                           0.82     10000
   macro avg       0.83      0.82      0.82     10000
weighted avg       0.83      0.82      0.82     10000
```
To see the other cross validation confusion matrix see this file, 
[Classification report AutoEncoder CNN model](https://github.com/shakhaout/cifar10_classification/blob/main/checkpoints/AutoEncoder_classification_report_noaug.txt)

# Train
To train the CNN model run below command:
```
python main.py --model_name CNN
```
To train the AutoEncoder CNN model run below command:
```
python main.py --model_name AUTOENCODER_CLS
```
To train the Modified AutoEncoder CNN model run below command:
```
python main.py --model_name MOD_AUTOENCODER_CLS
```
For the first time AutoEncoder CNN and Modified AutoEncoder CNN model will train the autoencoder model. After finished training the autoencoder model, using the weights of this autoencoder model CNN classification model will start training. If pretrained autoencoder weight already exists in the checkpoint directory, above two models will train the CNN classification part only using the pretrained autoencoder weights as input.

# Test
To test run below command:
```
python test.py --model_name autoencoder --model_path ./checkpoint/autoencoder_best_wgt.h5 --batch_size 128
```
For --model_name  use as follows, 
* AutoEncoder model = 'autoencoder'
* AutoEncoder CNN classification = 'autoencoder_cls'
* CNN classification = 'cnn'
