#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 01:44:30 2017

@author: davidjoy
"""
import tensorflow #machine learning  
import keras 
from keras.models import Sequential # backend tensor flow 
import numpy as np 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

#Step 1 - collect data 
img_width,img_height=150,150

train_data= '/Users/davidjoy/Desktop/Data'
validation_data = '/Users/davidjoy/Desktop/Data1'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

#Step 2 - Build Model 
model = Sequential()
model.add(Convolution2D(32, 3, 3,input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3),input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,(3,3),input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dropout(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Step 3 - Train Model 
nb_epoch = 2
nb_train_samples = 2308
nb_validation_samples = 881

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

model.evaluate_generator(validation_generator, nb_validation_samples)

# model.save_weights('models/simple_CNN.h5')

#Step 4 - Test Model 
from keras.preprocessing.image import load_img
img = load_img('/Users/davidjoy/Desktop/test/cat.181.jpg',target_size=(150,150))
img = np.array(img)
type(img)
img1 = img.reshape( (1,) + img.shape )
type(img1)
prediction = model.predict(img1)









print prediction


>> print(img.shape)
(5000, 32, 32, 3)
# we have 5000 examples, pixel 32X32, RGB color 3 channels
# but for KNN, we don't need this layered information, so we just flatten out the later 3 dimensions:
>> img1 = np.reshape(img, (img.shape[0], 1))
>> print(X_train.shape)
(5000, 3072)