# CIFAR-10 Image Classification using CNN Model and Pretrained AutoEncoder CNN Model 
Task is to design a network that combines supervised and unsupervised architectures in one model to achieve a classification task.

## Task Description
The model must start with autoencoder(s) (stacking autoencoder is ok) that is connected through its hidden layer to another network of choice, as shown in the figure below:
![alt text](https://github.com/shakhaout/cifar10_classification/blob/main/imgs/Problem_architecture.png)
This autoencoder takes an input (image) at node 1 and reconstructs it at its output at node 3. It creates valuable features at its hidden layers (node 2) during this process. it is hypothesized that if node 2 is used as input for the CNN (node 4) then the classification can be improved.
### Data:
We want to use this model to classify Cifar-10 dataset by using at max 2500 training images for each of the bird, deer, and truck classes while using 5000 for the other classes. The model should be evaluated by the test set of 10000 images (1000 for each class).

