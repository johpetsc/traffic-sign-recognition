# Traffic sign recognition
Detection and recognition of traffic signs.  
The recognition model was trained using fastai and the GTSRB dataset, with the best result of 99.99% accuracy for Resnet-50.  
The detection model was trained based on the GTSDB and a pre-trained object detection model from Tensorflow's [Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
 
The detection model is trained to recognize more general features, such as the sign shape and color, while the recognition model is the one in charge of differentiating the signs based on more specific features.

![Example](https://github.com/johpetsc/traffic-sign-recognition/blob/main/results/images/37.jpg?raw=true)
# Requirements
To run it locally, it's recommended to have at least 16GB of RAM and a GPU with 6GB of VRAM. If your computer does not meet the requirements, you can adapt the code for a Google Colab notebook.
- fastai 
- Python 3.7.x (or later)  
- OpenCV 4.4
- Pytorch
- numpy
- tensorflow 2.x
- [Object Detection API for TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)  
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- cudNN (Requires a NVIDIA Developer account do download)

# Execute

Convert the detection dataset to TFRecords:
```
    $ python src/create_tfrecord.py
```
Train the recognition model:
```
    $ python src/recognition.py
```
Train the detection model:
```
    $ python src/train_detection.py --model_dir=models/faster_rcnn --pipeline_config_path=models/faster_rcnn/pipeline.config
```
Export the detection model after training:
```
    $ python src/export_model.py --input_type image_tensor --pipeline_config_path .\models\faster_rcnn\pipeline.config --trained_checkpoint_dir .\models\faster_rcnn\ --output_directory .\models\detection
```
Detect and recognize traffic signs on test images:
```
    $ python src/detection.py
```

# Tutorials
- [How to train your own detection model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

# Dataset
- [GTSRB Recognition dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- [Recognition dataset with labeled folders](https://github.com/johpetsc/traffic-sign-recognition/tree/main/dataset)  
- [GTSDB Detection dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) 
- [Detection dataset converted to JPG](https://github.com/johpetsc/traffic-sign-recognition/tree/main/dataset/Training/GTSDB/train) 

