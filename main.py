import os
import sys
from PIL import Image
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vis import plot_data_distribution, loss_curve, accuracy_curve, show_org_rcnst_img
from utils import imbalanced_dataset, BalancedDataGenerator, PSNR_SSIM, report
from models import encoder, decoder, classifier, EnCNN_cls

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
os.getcwd()

###########################################################################################################################

## create directories
save_dir ='./checkpoint/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
img_dir = './imgs/'
if not os.path.isdir(img_dir):
    os.makedirs(img_dir)

    

###########################################################################################################################


class Classification:
    def __init__(self, x_train, y_train, X_test, Y_test, val_split, batch_size, epochs,lr,
                 patience, input_image = Input(shape=(32,32,3))):
        self.x_train = x_train
        self.y_train = y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.input_image = Input(shape=(32,32,3))

    def autoencoder(self):
        
        if os.path.isfile(save_dir+'autoencoder_best_wgt.h5') == False:
            datagen = ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
            X_train,X_val,Y_train,Y_val = train_test_split(self.x_train, self.y_train, test_size=self.val_split,
                                                           random_state=1, shuffle= True, stratify=self.y_train)
            steps = int(X_train.shape[0]/self.batch_size)
            early_stop = EarlyStopping(monitor='val_loss',patience=self.patience,verbose=1,mode='min')
            rlr=ReduceLROnPlateau(patience=10,monitor="val_loss",factor=0.3, min_lr=0.0001, verbose=1,cooldown=1)
            ckpt_path = save_dir+'autoencoder_best_wgt.h5'
            checkpoint = ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', verbose=1, save_best_only=True,
                                         mode='min')
            train_itr = datagen.flow(X_train, X_train, batch_size=self.batch_size) 
            validation_itr = datagen.flow(X_val, X_val, batch_size=self.batch_size)
            clf = Model(self.input_image,decoder(encoder(self.input_image)))
            # COMPILE NEW MODEL
            clf.compile(loss='mean_squared_error',optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])
            hist = clf.fit(train_itr,
                           steps_per_epoch = steps,
                           epochs = self.epochs,
                           verbose = 1,
                           validation_data = validation_itr,
                           callbacks = [early_stop, checkpoint],
                           shuffle =True
                                 )
            loss_curve('autoencoder',  hist)
            accuracy_curve('autoencoder',  hist)
            pred_x = clf.predict(self.X_test)
            show_org_rcnst_img(self.X_test,pred_x)
            # PSNR & SSIM
            PSNR_SSIM(self.X_test,clf)
            print(clf.summary())
            m_s= open(save_dir + './autoencoder_summary.txt','w')
            stringlist = []
            clf.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            m_s.write(model_summary)
            plot_model(clf, img_dir+'autoencoder_model_architecture.png', show_shapes=True)

        else:
            clf = load_model(save_dir+'autoencoder_best_wgt.h5')

        return clf
    
    def model_training(self,model_name=None,pretrained = None):

        fold_var = 1
        datagen = ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
        if model_name == 'CNN':
            REPORT = open(save_dir+'CNN_classification_report.txt','w')
        elif model_name == 'AUTOENCODER_CLS':
            REPORT = open(save_dir+'AutoEncoder_classification_report.txt','w')
        elif model_name == 'MOD_AUTOENCODER_CLS':
            REPORT = open(save_dir+'Modified_AutoEncoder_classification_report.txt','w')
        #Stratified ShuffleSplit cross-validator
        sss = StratifiedShuffleSplit(n_splits=5, test_size=self.val_split, random_state=1)

        for train_index, val_index in sss.split(self.x_train, self.y_train):
            print('KFold No:',fold_var)
            X_train, X_val = self.x_train[train_index], self.x_train[val_index]
            Y_train, Y_val = self.y_train[train_index], self.y_train[val_index]

            plot_data_distribution(Y_train,Y_val)
            # ONE HOT ENCODING
            Y_train = to_categorical(Y_train)
            Y_val = to_categorical(Y_val)
            if model_name == 'CNN':
                # CREATE NEW MODEL
                clf = Model(self.input_image,classifier(self.input_image))
                # COMPILE NEW MODEL
                clf.compile(loss = categorical_crossentropy,
                                     optimizer = Adam(lr=self.lr),
                                     metrics = ['accuracy'])
                plot_model(clf, img_dir+'CNN_classification_model_architecture.png', show_shapes=True)
                ckpt_path = save_dir+'CNN_classification_best_wgt_'+str(fold_var)+".h5"

            elif model_name == 'AUTOENCODER_CLS':
                encode = encoder(self.input_image) 
                clf = Model(self.input_image,classifier(encode))
                for l1, l2 in zip(clf.layers[0:11], pretrained.layers[0:11]):
                    l1.set_weights(l2.get_weights())
                for layer in clf.layers[0:11]:
                    layer.trainable = False
                clf.compile(loss = categorical_crossentropy,
                                     optimizer = Adam(lr=self.lr),
                                     metrics = ['accuracy'])
                plot_model(clf, img_dir+'AutoEncoder_classification_model_architecture.png', show_shapes=True)
                ckpt_path = save_dir+'AutoEncoder_classification_best_wgt_'+str(fold_var)+".h5"

            elif model_name == 'MOD_AUTOENCODER_CLS':
                encode = encoder(self.input_image) 
                clf = Model(self.input_image,EnCNN_cls(encode))
                for l1, l2 in zip(clf.layers[0:19], pretrained.layers[0:19]):
                    l1.set_weights(l2.get_weights())
                for layer in clf.layers[0:19]:
                    layer.trainable = False
                clf.compile(loss = categorical_crossentropy,
                                     optimizer = Adam(lr=self.lr),
                                     metrics = ['accuracy'])
                plot_model(clf, img_dir+'Modified_AutoEncoder_classification_model_architecture.png', show_shapes=True)
                ckpt_path = save_dir+'Modified_AutoEncoder_classification_best_wgt_'+str(fold_var)+".h5"

            # Model Callbacks
            steps = int(X_train.shape[0]/self.batch_size)
            early_stop = EarlyStopping(monitor='val_loss',patience=self.patience,verbose=1,mode='min')
            rlr=ReduceLROnPlateau(patience=10,monitor="val_loss",factor=0.3, min_lr=0.0001, verbose=1,cooldown=1)
            checkpoint = ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', verbose=1, save_best_only=True,
                                         mode='min')
            # Model Fit

            train_bgen = BalancedDataGenerator(X_train, Y_train, datagen, batch_size=self.batch_size)
            validation_bgen = BalancedDataGenerator(X_val, Y_val, datagen, batch_size=self.batch_size)
            hist = clf.fit(train_bgen,
                           steps_per_epoch = train_bgen.steps_per_epoch,
                           epochs = self.epochs,
                           verbose = 1,
                           validation_data = validation_bgen,
                           callbacks = [early_stop, rlr, checkpoint],
                           shuffle =True
                                 )


            if model_name == 'CNN':
                loss_curve('CNN_classification',  hist, fold_var)
                accuracy_curve('CNN_classification',  hist, fold_var)
                pred_y = clf.predict(self.X_test)
                REPORT.write('Kfold Iteration:')
                REPORT.write(str(fold_var))
                REPORT.write('\n')
                rpt = report(self.Y_test,pred_y,'CNN_classification',fold_var)
                REPORT.write(rpt)
                  
            elif model_name == 'AUTOENCODER_CLS':
                loss_curve('AutoEncoder_classification',  hist, fold_var)
                accuracy_curve('AutoEncoder_classification',  hist, fold_var)
                pred_y = clf.predict(self.X_test)
                REPORT.write('Kfold Iteration:')
                REPORT.write(str(fold_var))
                REPORT.write('\n')
                REPORT.write(report(self.Y_test,pred_y,'AutoEncoder_classification',fold_var))

            elif model_name == 'MOD_AUTOENCODER_CLS':
                loss_curve('Modified_AutoEncoder_classification',  hist, fold_var)
                accuracy_curve('Modified_AutoEncoder_classification',  hist, fold_var)
                pred_y = clf.predict(self.X_test)
                REPORT.write('Kfold Iteration:')
                REPORT.write(str(fold_var))
                REPORT.write('\n')
                REPORT.write(report(self.Y_test,pred_y,'Modified_AutoEncoder_classification',fold_var))
                 

            fold_var += 1

        print(clf.summary())
        stringlist = []
        clf.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        REPORT.write('\n')
        REPORT.write(model_summary)

        
#################################################################################################################

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_name', help='Input the Model Name as: CNN, AUTOENCODER_CLS, MOD_AUTOENCODER_CLS')
    parse.add_argument('--val_split', default=0.15, type = float, help='Input the split size of train validation set')
    parse.add_argument('--batch_size', default=128, type =int, help='Input the mini batch size')
    parse.add_argument('--epochs', default=100, type=int, help='Input the number of epochs to train')
    parse.add_argument('--lr', default=0.001, type= float, help='Input the Learning rate')
    parse.add_argument('--patience', default=15, type=int, help='Input the early stopping patience limit')
    args = parse.parse_args()

    x_train, y_train, X_test, Y_test = imbalanced_dataset()

    Train = Classification(x_train=x_train,
                            y_train=y_train,
                            X_test=X_test,
                            Y_test=Y_test,
                            val_split=args.val_split, 
                            batch_size=args.batch_size, 
                            epochs=args.epochs,
                            lr=args.lr,
                            patience=args.patience)
    if args.model_name == 'CNN':
        hist = Train.model_training(args.model_name)
        
    elif args.model_name == 'AUTOENCODER_CLS':
        encoder_decoder = Train.autoencoder()
        hist = Train.model_training(args.model_name, encoder_decoder)

    elif args.model_name == 'MOD_AUTOENCODER_CLS':
        encoder_decoder = Train.autoencoder()
        hist = Train.model_training(args.model_name, encoder_decoder)

    else:
        print('Wrong Argument, Input the Model Name as: "CNN", "AUTOENCODER_CLS" or "MOD_AUTOENCODER_CLS"! ')


