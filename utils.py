from tensorflow.keras.datasets import cifar10
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vis import plot_data_distribution
from tensorflow.keras.utils import to_categorical, Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator
from sklearn.metrics import classification_report, confusion_matrix
from skimage import metrics
#########################################################################################

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 
                  5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

## create imbalanced dataset taking 2500 data for bird, deer and truck
def imbalanced_dataset(data= cifar10.load_data(), imbl_size=2500,val_split=0.15):
    (x_trn,y_trn),(x_test,y_test) = data
    x_bird = []
    y_bird = []
    x_deer = []
    y_deer = []
    x_truck = []
    y_truck = []
    x_train = []
    y_train = []
    for i,j in zip(x_trn,y_trn):
        if j == 2:
            x_bird.append(i)
            y_bird.append(j)
        elif j == 4:
            x_deer.append(i)
            y_deer.append(j)
        elif j ==9:
            x_truck.append(i)
            y_truck.append(j)
        else:
            x_train.append(i)
            y_train.append(j)
    bird_2500 = random.sample(x_bird,imbl_size)
    for i,j in zip(bird_2500,y_bird[:imbl_size]):
        x_train.append(i)
        y_train.append(j)
    deer_2500 = random.sample(x_deer,imbl_size)
    for i,j in zip(deer_2500,y_deer[:imbl_size]):
        x_train.append(i)
        y_train.append(j)
    truck_2500 = random.sample(x_truck,imbl_size)
    for i,j in zip(truck_2500,y_truck[:imbl_size]):
        x_train.append(i)
        y_train.append(j)
        
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    # Normalizing the dataset
    x_train = x_train.astype('float32')
    x_train = x_train/255.0
    x_test = x_test.astype('float32')
    x_test = x_test/255.0
    
    # One Hot Encoding
    y_test = to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test
#######################################################################################################

## function used for creating a balanced mini batch augmented image generator
class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=64):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * self.batch_size, *x.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
###########################################################################################################

## function used for calculating Peak signal to noise ratio and structural index similarity
def PSNR_SSIM(X_test,model):
    avg_psnr=0
    avg_ssim=0
    test_size=0
    for data in X_test:
        img= np.expand_dims(data, axis=0)
        output = model.predict(img)
        org_img = np.squeeze(img, axis=0)
        out_img = np.squeeze(output,axis=0)
        avg_psnr+=metrics.peak_signal_noise_ratio(org_img,out_img)
        avg_ssim+=metrics.structural_similarity(org_img,out_img,multichannel=True)
        test_size+=len(img)

    print("On Test data of {} examples:\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(test_size,avg_psnr/test_size,avg_ssim/test_size))
###########################################################################################################

## function used for creating a classification report and confusion matrix
def report(Y_test,predictions, model, fold_var): 
    cm=confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Classification Report:\n")
    cr=classification_report(Y_test.argmax(axis=1),
                                predictions.argmax(axis=1), 
                                target_names=list(label_dict.values()))
    print(cr)
    plt.figure(figsize=(12,12))
    sns_plot = sns.heatmap(cm, annot=True, xticklabels = list(label_dict.values()), 
                yticklabels = list(label_dict.values()), fmt=".2f")
    
    if fold_var == None:
        sns_plot.figure.savefig('./imgs/{}_model_heatmap.png'.format(model),bbox_inches='tight')
        Title = '{} model heatmap'.format(model)
        sns_plot.set_title(Title)
        plt.show()
    else:
        Title = '{} model Kfold {} heatmap'.format(model,fold_var)
        sns_plot.set_title(Title)
        sns_plot.figure.savefig('./imgs/{}_model_kfold{}_heatmap.png'.format(model,fold_var),bbox_inches='tight')
    return cr

