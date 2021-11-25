import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
import glob as glob
import recognition as rec
import cv2 as cv

PATH_TO_MODEL = 'models/detection/saved_model' #Path to the pre-trained model
PATH_TO_LABELS = 'models/annotations/label_map.pbtxt'
TEST_IMAGES = glob.glob('test_images/*.jpg') #Path to test images

def visualize_detections(image, detections):
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    cv.imshow("img", image)#Shows image
    cv.waitKey(0)

def image_to_numpy(image):
    image = Image.open(image)
    w, h = image.size
    image_np = np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8) #Image to numpy array

    return image_np, w, h

def numpy_to_tensor(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = np.expand_dims(image_np, 0)

    return input_tensor

def detection(model, tensor):
    detections = model(tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

def filter_detections(detections, w, h):
    boxes = 0 # number of valid detections
    filtered_detections = []
    for box, score in zip(np.squeeze(detections['detection_boxes']), detections['detection_scores']): #Bounding box and score from each detection
        if score*100 > 20 or boxes < 7: #>20%
            box[0], box[1], box[2], box[3] = box[0]*h, box[1]*w, box[2]*h, box[3]*w #Normalized bounding boxes coordinates
            filtered_detections.append(box) #Append to list with filtered detections
            boxes += 1
        else:
            break
    
    return filtered_detections, boxes

def main():
    recognition_model = rec.getModel()
    detection_model = tf.saved_model.load(PATH_TO_MODEL)

    for image in TEST_IMAGES: #For each image in the test images folder
        image_np, w, h = image_to_numpy(image)
        input_tensor = numpy_to_tensor(image_np)

        detections = detection(detection_model, input_tensor)
        filtered_detections, boxes = filter_detections(detections, w, h)
        rec.predict(image_np, recognition_model, filtered_detections, boxes)
        # visualize_detections(image_np.copy(), detections)

if __name__ == '__main__':
    main()