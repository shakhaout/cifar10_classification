import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

################################################################################################################

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 
                  5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


def plot_data_distribution(y_train,y_test):
    
    # Visualize the data distribution
    trn_data_y = np.concatenate(y_train).ravel()
    # val_data_y = np.concatenate(Y_val).ravel()
    tst_data_y = np.concatenate(y_test)
    train_count = Counter(trn_data_y)
    # val_count = Counter(val_data_y)
    test_count = Counter(tst_data_y)
    
    fig,ax = plt.subplots(ncols=2,figsize=(50,15))
    values = [train_count, test_count]
    Titles = ['Train', 'Validation']
    for i in range(len(values)):
        ax[i].bar(values[i].keys(), values[i].values())
        ticks= [label_dict[x] for x in sorted(values[i].keys())]
        ax[i].set_xticks(np.arange(len(values[i].keys())))
        ax[i].set_xticklabels(ticks)
        tle = "{} Data Distribution".format(Titles[i])
        ax[i].set_title(tle)
    fig.savefig("./imgs/data_distribution.png",bbox_inches='tight')
#     plt.show()

###############################################################################################################
    
def loss_curve(model_name=None,hist=None,fold_var=None):
    plt.figure(figsize=(12,7))
    plt.title('Loss plot of {} model'.format(model_name))
    plt.plot(hist.history['loss'], color='blue', label='train')
    plt.plot(hist.history['val_loss'], color='orange', label='validation')
    plt.legend()
    # save plot to file
    if fold_var == None:
        plt.savefig('./imgs/loss_plot_{}.png'.format(model_name),bbox_inches = 'tight')
    else:
        plt.savefig('./imgs/loss_plot_{}_kfold{}.png'.format(model_name,fold_var),bbox_inches = 'tight')
#     plt.show()
    
#############################################################################################################

def accuracy_curve(model_name=None,hist=None,fold_var=None):
    plt.figure(figsize=(12,7))
    plt.title('Accuracy plot of {} model'.format(model_name))
    plt.plot(hist.history['accuracy'], color='blue', label='train')
    plt.plot(hist.history['val_accuracy'], color='orange', label='validation')
    plt.legend()
    # save plot to file
    if fold_var == None:
        plt.savefig('./imgs/accuray_plot_{}.png'.format(model_name),bbox_inches = 'tight')
    else:
        plt.savefig('./imgs/accuracy_plot_{}_kfold{}.png'.format(model_name,fold_var),bbox_inches = 'tight')
#     plt.show()
    
#######################################################################################################################

def show_org_rcnst_img(orig, dec, num=10, show=None):  ## function used for visualizing original and reconstructed images of the autoencoder model
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(dec[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="blue")
    plt.figtext(0.5,0.5, "RECONSTRUCTED IMAGES", ha="center", va="top", fontsize=14, color="blue")
    plt.subplots_adjust(hspace = 0.3 )
    ax.figure.savefig('./imgs/autoencoder_org_reconstd_imgs.png',bbox_inches='tight')
    if show != None:
        plt.show()