import numpy as np
import matplotlib.pyplot as plt
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *
import cv2 as cv

#Function fo training
def train():
    signs = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42), #20% of the images for validation
        get_y=parent_label, #Folder name as label
        item_tfms=Resize(128))

    dls = signs.dataloaders('dataset/Training/GTSRB/Final_Training/Images', num_workers=0, bs=16) #Load the Training dataset with batch size of 16 (higher means faster but less accurate)
    dls.valid.show_batch(max_n=15, nrows=3)#Show batch
    plt.show()#Not needed if running on Jupyter or Collab

    #Train the model
    learn = cnn_learner(dls, resnet50, metrics=[accuracy, error_rate])#Learning parameters, resnet50 got the best results, but resnet34 is also good. No improvement noted with resnet101+
    learn.model.cuda()#Use this if fastai is using your CPU instead of the GPU
    learn.fine_tune(5)#Number of iterations, more than 6 is unnecessary
    learn.recorder.plot_loss()#Shows valid and train loss
    plt.show()

    #Show confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(13,13))
    plt.show()

    #Show top losses
    interp.plot_top_losses(20, nrows=5)
    plt.show()

    #Export the trained model
    learn.export('models/recognition/export.pkl')

#Function for predictions.
#Parameters: image, the recognition model and a list of detection boxes.
def predict(img, learn_inf, boxes, num_detections):
    image = cv.cvtColor(img, cv.COLOR_RGB2BGR)#Transforming image from RGB to BGR because of OpenCV
    for box in boxes: #Each detection box in the list
        yi, yf, xi, xf = int(box[0]), int(box[2]), int(box[1]), int(box[3])#bounding box coordinates in the image
        sign = img[yi:yf, xi:xf]#bounding box content to variable
        pred, pred_idx, probs = learn_inf.predict(sign)#prediction
        output = pred + '[' + str(round(float(probs[pred_idx]*100))) + '%]'#Prediction label and probability from tensor to rounded float to string
        if num_detections > 0 and probs[pred_idx]*100 > 60:#Gets the yield sign from the rotated image detection
            print('Prediction: ', output)
            cv.rectangle(image, (xi, yi), (xf, yf), (0, 0, 0), 5)#Draws the bounding box, white rectangle with black borders
            cv.rectangle(image, (xi, yi), (xf, yf), (255, 255, 255), 3)
            cv.putText(image, output, (xi-40, yi), 2, 1, (0, 0, 0), lineType=cv.LINE_AA, thickness=2)#Draws the prediction, white text with black borders
            cv.putText(image, output, (xi-40, yi), 2, 1, (255, 255, 255), lineType=cv.LINE_AA, thickness=1)
            num_detections -= 1
    
    cv.imshow("img", image)#Shows image
    if cv.waitKey(0) & 0xFF == ord('q'):#Press Q to exit
        cv.imwrite('results/images/test/teste.jpg', image)

#Function used to load the already trained model
def getModel():
    learn_inf = load_learner('models/recognition/export.pkl')

    return learn_inf

if __name__ == '__main__':
    #Train the model
    train()