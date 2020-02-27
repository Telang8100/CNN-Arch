# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:09:15 2019

@author: MONSTER VISION
"""

import tensorflow as tf
tf.enable_eager_execution()
#AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import matplotlib.pyplot as plt
import random
#import os

data_root_orig ='./flower_photos/'
data_root = pathlib.Path(data_root_orig)#Reads the all subfolder paths in it.
print(data_root)

for item in data_root.iterdir():
  print(item)
  
all_image_paths = list(data_root.glob('*/*'))#list out the subfolders windows  paths in root directory
print(all_image_paths[:10])
all_image_paths = [str(path) for path in all_image_paths]#converts windows paths onto string
print(all_image_paths[:10])
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
#ds=all_image_paths[:10]
#print(ds)


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())#reads the subfolder names presents in the directory and 
                                                                                    #assign them as a labels 
print("\n",label_names)

label_to_index = dict((name, index) for index,name in enumerate(label_names))#Here the labels are given index 
                                                                            #i.e. The subfolder's names are conisderd as labels and indexed as 0,1,2...
print("\n",label_to_index)


all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 2 labels indices: ", all_image_labels[:2])


def load_and_preprocess_from_path_label(path, label):
  image = tf.read_file(path)  
  image = tf.image.decode_jpeg(image, channels=3)#Decode a JPEG-encoded image to a uint8 tensor.
                                                  #Returns:- A Tensor of type uint8.

  image = tf.image.resize(image, [500,333])#Resize images to size using the specified method.
                                              #Returns:-If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
                                              #If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]
  image /= 255.0  # normalize to [0,1] range

  return image,label

#def load_and_preprocess_image(path):
#  image = tf.read_file(path)#Reads and outputs the entire contents of the input filename.
#                              #Return:- A Tensor of type string
#  return preprocess_image(image)


## The tuples are unpacked into the positional arguments of the mapped function
#def load_and_preprocess_from_path_label(path, label):
#  return preprocess_image(path), label


#Here we combined images and labels in single statement

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))#Gives the dataset containing images and its corresponding labels.



BATCH_SIZE = 2


image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)

ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
#ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)


def model_alexnet():
    input_layer=tf.keras.layers.Input(shape=(500,333,3))
    #Layer1
    x=tf.keras.layers.Conv2D(96,(11,11),strides=4,activation='relu')(input_layer)
    x=tf.keras.layers.MaxPooling2D((3,3),strides=2)(x)
    #layer 2
    x=tf.keras.layers.Conv2D(256,(5,5),padding='same',strides=1,activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D((3,3),strides=2)(x)
    #Layer 3
    x=tf.keras.layers.Conv2D(384,(3,3),padding='same',strides=1,activation='relu')(x)
    #Layer 4
    x=tf.keras.layers.Conv2D(384,(3,3),padding='same',strides=1,activation='relu')(x)
    #Layer 5
    x=tf.keras.layers.Conv2D(256,(3,3),padding='same',strides=1,activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D((3,3),strides=2)(x)
    #Layer 6
    x=tf.keras.layers.Flatten()(x)
    #Layer 7
    x=tf.keras.layers.Dense(4096,activation='relu')(x)
    x=tf.keras.layers.Dropout(rate=0.5)(x)
    #Layer 8
    x=tf.keras.layers.Dense(4096,activation='relu')(x)
    x=tf.keras.layers.Dropout(rate=0.5)(x)
    #Layer 9(Softmax Layer)
    out=tf.keras.layers.Dense(5,activation='softmax')(x)
    
    model=tf.keras.Model(inputs=[input_layer],outputs=[out])
    return model

alexnet=model_alexnet()
alexnet.summary()

alexnet.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

mod=alexnet.fit(ds,epochs=3,steps_per_epoch=10)

acc=mod.history['acc']

print("Accuracy after Last Epoch is ",acc[-1])

#mod.evaluate()




