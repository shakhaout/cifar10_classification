import os
import argparse
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from utils import imbalanced_dataset, PSNR_SSIM, report
from vis import show_org_rcnst_img


parse = argparse.ArgumentParser()
parse.add_argument('--model_name', help='Input the Model name as autoencoder, autoencoder_cls, cnn')
parse.add_argument('--model_path', help='Input the saved weight path')
parse.add_argument('--batch_size', help='Input the mini batch size')
args = parse.parse_args()

x_train, y_train, X_test, Y_test = imbalanced_dataset()
def test_model(model_name=None,model_path=None, batch_size=128):
    if model_name == 'autoencoder':
        print(os.path.join(os.getcwd(),model_path))
        clf = load_model(os.path.join(os.getcwd(),model_path))
        pred_x = clf.predict(X_test)
        show_org_rcnst_img(X_test,pred_x,show=True)
        # PSNR & SSIM
        PSNR_SSIM(X_test,clf)
    else:
        clf = load_model(os.path.join(os.getcwd(),model_path))
        pred_y = clf.predict(X_test)
        rpt = report(Y_test,pred_y,model_name,None)
    
test_model(args.model_name, args.model_path,args.batch_size)
