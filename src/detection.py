import numpy as np
import tensorflow as tf
from PIL import Image
import glob as glob
import recognition as rec

PATH_TO_MODEL = '../models/detection/frozen_inference_graph.pb'
PATH_TO_TEST_IMAGES = glob.glob('../test_images/*.jpg')

def detection(image_np, detection_graph, sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores

def filter_detections(detected, boxes, scores, w, h):
    for box, score in zip(np.squeeze(boxes), np.squeeze(scores)):
        if score*100 > 20: 
            box[0], box[1], box[2], box[3] = box[0]*h, box[1]*w, box[2]*h, box[3]*w
            detected.append(box)
    return detected

def filter_detections_rotated(detected, boxes, scores, w, h):
    for box, score in zip(np.squeeze(boxes), np.squeeze(scores)):
        if score*100 > 20: 
            box[0], box[1], box[2], box[3] = h-(box[2]*h), w-(box[3]*w), h-(box[0]*h), w-(box[1]*w)
            detected.append(box)
    return detected

def main():
    detection_graph = tf.Graph()
    learn_inf = rec.getModel()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        fid = tf.io.gfile.GFile(PATH_TO_MODEL, 'rb')
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph, config=config)

        for image_path in PATH_TO_TEST_IMAGES:
            image = Image.open(image_path)
            w, h = image.size
            detected = []

            image_np = np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)
            image_np_r = np.array(image.rotate(180).getdata()).reshape((h, w, 3)).astype(np.uint8)

            boxes, scores = detection(image_np, detection_graph, sess)
            detected = filter_detections(detected, boxes, scores, w, h)
            
            boxes, scores = detection(image_np_r, detection_graph, sess)
            detected = filter_detections_rotated(detected, boxes, scores, w, h)
                    
            rec.predict(image_np, learn_inf, detected)

if __name__ == '__main__':
    main()