# facevec

A library for face detection and vectorisation

## Setup
Download respository and start setup.cmd

## Detection
For detection a Convolution Neural Network is trained to create bounding boxes. After the neural Network Non Max Supression is applied to remove low scored bounding boxes. The Neural Network can run up to 25 fps on an Intel i7-1255U as an onnx model. 

![plot](/images/detector.png)

## Face Vectorisation
To compare two faces the face has to be converted into a vector. With this vectoriser faces can be converted into a ( -1, 1024 ) vector.

![plot](/images/vectoriser.png)

## Face Point Detection

### Implementation of the algorithms
Neural Network to detect 68 face keypoints. To achieve better results add to the detected face width and height 0.1 % to it. 

![plot](/images/facePointDetector.png)

### Face point indexes
![plot](/images/FacePoints.png)
