# Traffic sign recognition
Detection and recognition of traffic signs on public roads.  
The recognition model was trained using fastai and the GTSRB dataset, with the best result of 99.99% accuracy for Resnet-50.  
To get the bounding boxes coordinates, the code uses a pre-trained model which you can find in [this](https://github.com/aarcosg/traffic-sign-detection#pretrained-models) repository.  
The detected signs are then classified using the recognition model.  

![Example](https://github.com/johpetsc/traffic-sign-recognition/blob/main/results/images/37.jpg?raw=true)
# Requirements
If you want to run it locally, it's recommended to have at least 16GB of RAM and a GPU with 6GB of VRAM. If your computer does not meet the requirements, you can adapt the code for a Google Colab notebook.
- fastai 2.1.7  
- Python 3.7.x (or later)  
- OpenCV 4.4.0  
- Pytorch 1.7.0  
- matplotlib 3.3.3  
- numpy 1.19.3  
- tesnorflow 2.2.0  
- [Object Detection API with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)  
- CUDA 10.0.0 or 10.0.1 (Depends on your Tensorflow version, if it asks for dynamic library x_10.dll, it's 10.0.0, if it asks for x_101.dll, it's 10.0.1)  
- cudNN 7  

# Dataset
- [Recognition dataset can be found here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), [or with labeled folders in this repository.](https://github.com/johpetsc/traffic-sign-recognition/tree/main/dataset)  
- [Detection dataset can be found here](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) *Not needed, dowload the pre-trained model below. Can be used for more test images.  
- [Recognition models, one was trained with resnet34 and the other with resnet50](https://github.com/johpetsc/traffic-sign-recognition/tree/main/models/recognition)  
- [Detection model (direct download)](https://drive.google.com/open?id=12vLvA9wyJ9lRuDl9H9Tls0z5jsX0I0Da), [repository with pre-trained models for detection](https://github.com/aarcosg/traffic-sign-detection#pretrained-models)  

# Running the code
1. Download the detection model and extract the frozen graph to models/detection
2. Download the recognition model from this repository or run python src/recognition.py to train a new model
3. Run python src/detection.py to detect and classify traffic signs from the test images.
