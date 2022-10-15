import sys
import os
import time
import tensorflow as tf
import keras
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from turtle import pd
from keras.backend import _regular_normalize_batch_in_training
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.recurrent_v2 import LSTM
from keras.preprocessing import image 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
from matplotlib.cbook import flatten
from sklearn import metrics
from keras.utils import np_utils
from keras.preprocessing import image as image_utils
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from tensorflow.python.client import device_lib
from keras.regularizers import l2


classes = {'fist' : 0, 'index' : 1, 'palm' : 2, 'peace' : 3}
#classes = {'fist' : 0, 'one' : 1, 'palm' : 2, 'two' : 3}

X_train = [] #images
Y_train = [] #labels
X_test = [] 
Y_test = [] 

def plot_image(path):
  img = cv2.imread(path) 
  img = cv2.resize(img, (100, 100))
  plt.imshow(img, cmap="gray") 
  plt.xlabel("Width")
  plt.ylabel("Height")
  plt.title("Image " + path)
  plt.show()


#plot_image(imagepaths[10]) 

def training():

  global X_train, X_test, Y_train, Y_test

  imagepaths = []
  #saving image paths in a list
  for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
      path = os.path.join(root, name)
      if path.endswith("png") and ('test' in path or 'train' in path) and ('data' in path): 
        imagepaths.append(path)
  #print(len(imagepaths))

  for path in imagepaths:
    img = cv2.imread(path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    category = path.split('\\')
    
    key = category[3].split("_")[0]
    if key in classes.keys():
      label = classes[key]

    if category[2]=="train":
      X_train.append(img)
      Y_train.append(label)
    elif category[2]=="test":
      X_test.append(img)
      Y_test.append(label)

  dataset = list(zip(X_train, Y_train))
  random.shuffle(dataset)
  dataset2 = list(zip(X_test, Y_test))
  random.shuffle(dataset2)

  
  X_train = np.array([d[0] for d in dataset], dtype="uint8")
  Y_train = np.array([d[1] for d in dataset])
  X_test = np.array([d[0] for d in dataset2], dtype="uint8")
  Y_test = np.array([d[1] for d in dataset2])

  Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes=4,dtype='i1')
  Y_test = keras.utils.np_utils.to_categorical(Y_test, num_classes=4,dtype='i1')

  print(X_train.shape)
  print(Y_train.shape)
  print(X_test.shape)
  print(Y_test.shape)

def classifying():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(89, 100, 1), padding="valid")) 
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4, activation='softmax'))

  e = 4
  bs = 50
  # Configures the model for training
  opt = tf.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  checkpoint = ModelCheckpoint("model1.h5",
                              monitor="accuracy",
                              mode="max",
                              save_best_only = True,
                              verbose=1)

  earlystop = EarlyStopping(monitor = 'accuracy', #Value being monitored for improvement
                            min_delta = 0.1, #Abs value and is the min change required before we stop
                            patience = 3, #Number of epochs we wait before stopping 
                            verbose = 1,
                            restore_best_weights = True) #Keeps the best weigths once stopped
  callbacks = [earlystop, checkpoint]

  # Trains the model for a given number of epochs and validates it.
  history = model.fit(X_train, Y_train, epochs=e, batch_size=bs, verbose=2, callbacks=callbacks, validation_data=(X_test, Y_test))


  train_loss, train_acc = model.evaluate(X_train, Y_train)
  print('Train accuracy: {:2.2f}%'.format(train_acc*100))

  test_loss, test_acc = model.evaluate(X_test, Y_test)
  print('Test accuracy: {:2.2f}%'.format(test_acc*100))

  model.save('classification_model.h5')

  plot_metrics(history, e, bs)
  #print(model.summary())
  #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def plot_metrics(history, e, bs):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  #plt.savefig('e' + str(e) + 'bs' + str(bs) + '.png')
  plt.show()


if __name__ == "__main__":


  training()
  classifying()

  classify = load_model('model1.h5')


  imagepaths = []
  for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
      path = os.path.join(root, name)
      if path.endswith("png") and ('trn' in path): 
        imagepaths.append(path)


  for path in imagepaths:
      img = Image.open(path)
      img = image.img_to_array(img)
      img = np.expand_dims(img, axis = 0)
      x_data = np.array(img, dtype = 'float32')
      x_data = x_data.reshape((len(x_data), 89, 100, 1))
      prediction = np.argmax(classify.predict(x_data))
      key = next(key for key, value in classes.items() if value == prediction)
      print(str(prediction) + str(key))
  
  #start = time.time()
  #model.predict() # Training statement
  #print("Time for prediction: ", time.time() - start, "seconds")




