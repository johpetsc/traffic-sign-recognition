# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import hashlib
import io
import glob
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import pandas as pd

PATH_TO_IMAGES = 'dataset/Training/GTSDB/train'
PATH_TO_LABELS = 'models/annotations/label_map.pbtxt'
IMAGE_LIST = glob.glob('dataset/Training/GTSDB/train/*.jpg')

def get_label_map(label):
    red_circle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 41, 42]  # circular, red, white
    blue_circle = [33, 34, 35, 36, 37, 38, 39, 40]  # circular, blue
    triangle = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # triangular, red, white
    ud_triangle = [13] # upsidedown triangle (yield sign)
    octagon = [14] # octagon, red (stop sign)
    square = [12] # rotated square

    if label in red_circle:
        return 1
    elif label in blue_circle:
        return 2
    elif label in triangle:
        return 3
    elif label in ud_triangle:
        return 4
    elif label in octagon:
        return 5
    elif label in square:
        return 6

def df_to_tf_example(data):
    label_map_dict = label_map_util.get_label_map_dict(PATH_TO_LABELS)
    with tf.compat.v1.gfile.GFile(PATH_TO_IMAGES + "/" + data['filename'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin, ymin, xmax, ymax = [], [], [], []
    classes, classes_text = [], []
    
    label_map_dict = {v: k for k, v in label_map_dict.items()}
    for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = label_map_dict[obj['class']]
        classes_text.append(class_name.encode('utf8'))
        classes.append(obj['class'])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def create_tf_record(output_filename, images):

    writer = tf.compat.v1.python_io.TFRecordWriter(output_filename)

    df = pd.read_csv(PATH_TO_IMAGES + '/gt.txt', delimiter=';', names=('file', 'xMin', 'yMin', 'xMax', 'yMax', 'classId'))
    df['file'] = df['file'].str.replace('ppm', 'jpg')

    for idx, image in enumerate(images):
        data = {'filename': image[-9:], 'object': []}
        signs = df[df['file'] == image[-9:]]
        for _, obj in signs.iterrows():
            class_id = get_label_map(obj['classId'])
            if class_id != -1:
                data['object'].append({'bndbox': {
                                        'xmin': obj['xMin'],
                                        'ymin': obj['yMin'],
                                        'xmax': obj['xMax'],
                                        'ymax': obj['yMax'] },
                                        'class': class_id})

        tf_example = df_to_tf_example(data)
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    create_tf_record('models/annotations/train.record', IMAGE_LIST[:700])
    create_tf_record('models/annotations/test.record', IMAGE_LIST[700:])

if __name__ == '__main__':
    tf.compat.v1.app.run()