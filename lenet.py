import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import random
import cv2
import pickle

np.random.seed(0)
# read pickeled data files
with open('german-traffic-signs/train.p', 'rb') as file:
    train_data = pickle.load(file)
with open('german-traffic-signs/valid.p', 'rb') as file:
    val_data = pickle.load(file)
with open('german-traffic-signs/test.p', 'rb') as file:
    test_data = pickle.load(file)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# check images and labels are same
assert(X_train.shape[0] == y_train.shape[0]), "Num train images != num images"
assert(X_val.shape[0] == y_val.shape[0]), "Num val images != num images"
assert(X_test.shape[0] == y_test.shape[0]), "Num train images != num images"
assert(X_train.shape[1:] == (32, 32, 3)), "Dimension of train images are not 32x32x3"
assert(X_val.shape[1:] == (32, 32, 3)), "Dimension of val images are not 32x32x3"
assert(X_test.shape[1:] == (32, 32, 3)), "Dimension of test images are not 32x32x3"

# pre processing
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

datagen = ImageDataGenerator(width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10)

datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

def modifiedModel():
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(Adam(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predictClass(model, img):
    predictions = model.predict(img)
    predictedClass = predictions.argmax(axis=-1)
    return predictedClass

def trainModel(model):
    print(model.summary())
    history = model.fit(datagen.flow(X_train, y_train, batch_size=50),
        steps_per_epoch=2000,epochs=10,
        validation_data=(X_val, y_val),shuffle=True)
    return history
