import numpy as np
import tensorflow as tf
from PIL import Image
import glob as glob
import recognition as rec

PATH_TO_MODEL = 'models/detection/frozen_inference_graph.pb' #Path to the pre-trained model
PATH_TO_TEST_IMAGES = glob.glob('test_images/*.jpg') #Path to test images

#Gets an image, detection graph and session, returns a list of detections and scores
def detection(image_np, detection_graph, sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    #Gets the detection graph tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    #Executes the detection
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores

#Filters detections with less than 20% probability
def filter_detections(detected, boxes, scores, w, h):
    count = 0 #Amount of valid detections
    for box, score in zip(np.squeeze(boxes), np.squeeze(scores)): #Bounding box and score from each detection
        if score*100 > 20: #>20%
            box[0], box[1], box[2], box[3] = box[0]*h, box[1]*w, box[2]*h, box[3]*w #Normalized bounding boxes coordinates
            detected.append(box) #Append to list with filtered detections
            count += 1
    return detected, count

#Filters detections with less than 20% probability in the rotated image
def filter_detections_rotated(detected, boxes, scores, w, h):
    for box, score in zip(np.squeeze(boxes), np.squeeze(scores)): #Bounding box and score from each detection
        if score*100 > 20: #>20%
            box[0], box[1], box[2], box[3] = h-(box[2]*h), w-(box[3]*w), h-(box[0]*h), w-(box[1]*w) #Normalized bounding boxes coordinates and transform to the real coordinates
            detected.append(box) #Append to list with filtered detections
    return detected

def main():
    detection_graph = tf.Graph()
    learn_inf = rec.getModel() #Loads the recognition model

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with detection_graph.as_default(): #Loads the detection model (frozen graph)
        od_graph_def = tf.compat.v1.GraphDef()
        fid = tf.io.gfile.GFile(PATH_TO_MODEL, 'rb')
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph, config=config) #Initializes the session
        for image_path in PATH_TO_TEST_IMAGES: #For each image in the test images folder
            image = Image.open(image_path)
            w, h = image.size
            detected = []

            image_np = np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8) #Image to numpy array
            image_np_r = np.array(image.rotate(180).getdata()).reshape((h, w, 3)).astype(np.uint8) #Rotated image to numpy array

            #Detect and filter detections for the normal image
            boxes, scores = detection(image_np, detection_graph, sess)
            detected, num_detections = filter_detections(detected, boxes, scores, w, h)
            
            #Detect and filter detections for the rotated image
            boxes, scores = detection(image_np_r, detection_graph, sess)
            detected = filter_detections_rotated(detected, boxes, scores, w, h)

            #Predicts which traffic sign each detection is      
            rec.predict(image_np, learn_inf, detected, num_detections)

if __name__ == '__main__':
    main()