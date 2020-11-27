import numpy as np
import readTrafficSigns as rts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fastai.data.all import *
from fastai.vision.all import *
import torch
import cv2 as cv

def train():
    signs = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    dls = signs.dataloaders('../dataset/Training/GTSRB/Final_Training/Images', num_workers=0, bs=16)
    dls.valid.show_batch(max_n=15, nrows=3)
    plt.show()

    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.model.cuda()
    learn.fine_tune(5)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.show()

    interp.plot_top_losses(15, nrows=3)
    plt.show()

    learn.export('../models/recognition/export.pkl')

def predict(img, learn_inf, boxes):
    image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for box in boxes:
        yi, yf, xi, xf = int(box[0]), int(box[2]), int(box[1]), int(box[3])
        sign = img[yi:yf, xi:xf]
        pred, pred_idx, probs = learn_inf.predict(sign)
        print('Prediction: ', pred, 'Probability: %5.2f' % (probs[pred_idx]*100), '%')
        cv.rectangle(image, (xi, yi), (xf, yf), (0, 0, 0), 5)
        cv.rectangle(image, (xi, yi), (xf, yf), (255, 255, 255), 3)
        cv.putText(image, pred, (xi-40, yi), 4, 1, (0, 0, 0), lineType=cv.LINE_AA, thickness=4)
        cv.putText(image, pred, (xi-40, yi), 4, 1, (255, 255, 255), lineType=cv.LINE_AA, thickness=2)
    
    cv.imshow("img", image)
    cv.waitKey(0)

def getModel():
    learn_inf = load_learner('export.pkl')

    return learn_inf

if __name__ == '__main__':
    train()
    #learn_inf = getModel()
    #predict('../data/shapes.jpg', learn_inf)