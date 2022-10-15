# Gesture Recognition for Remote Control of Photo Visualization Applications

This project creates an interface that uses hand gesture recognition to remotely navigate through image visualization applications. The intended focus of this system is towards aiding medical practitioners in browsing patient records and medical images without having to phyisically control a mouse or keyboard to combat sterility and inconvenience issues as posed by them. It utilizes four gestures, Fist, Index, Peace and Palm, which perform operations on the images including Zooming in and out, Pointing to specific parts of an image, Rotating the image by 90 degree, and Swiping through a set of images respectively.

It consists of 3 executionable files :
- capture.py is the main file to be run, that captures a live camera feed of the user performing hand gestures, predicts the gesture being performed, and executes the corresponding command onto an active Photo Viewer or Microsoft Photos window. 
- model.py is the file used to train and validate the classification model that carries out the gesture perdictions. Running this file will create the CNN model which will be stored as classification_model.h5 in this directory.
- generate.py can be used to create and save a dataset of new gestures for the training of the model.

This project is compatible with Windows and the Photo Viewer platform, and could also be compatible with other image viewing platforms with similar functionalities. Python 3.8+ is recommended to run this code.
Optimal results can be achieved by performing specified hand gestures on a pain dark background in the left half of the user live-feed window after selecting the appropriate colour threshold values for the detection of the hand. Also keep a photo opened in the Photo Viewer window to implement the commands onto, and activate or click onto the window to select it for command execution. 
