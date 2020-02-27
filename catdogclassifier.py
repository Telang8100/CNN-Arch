# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:32:44 2019

@author: MONSTER VISION
"""
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print(len(os.listdir('C:/Users/MONSTER VISION/Desktop/data/catdog/training_set/cats/')))
print(len(os.listdir('C:/Users/MONSTER VISION/Desktop/data/catdog/training_set/dogs/')))
print(len(os.listdir('C:/Users/MONSTER VISION/Desktop/data/catdog/test_set/cats/')))
print(len(os.listdir('C:/Users/MONSTER VISION/Desktop/data/catdog/test_set/dogs/')))

TRAINING_DIR = "C:/Users/MONSTER VISION/Desktop/data/catdog/training_set/"
train_datagen = ImageDataGenerator(rescale=1./255, 
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=2, 
                                                    class_mode='categorical',target_size=(374,374)) 
#flow_from_directory returns the images with labels 
# that labels are names to the folders

label_map = (train_generator.class_indices)#train_generator.clas_indices returns the labels in label_map variable.
print(label_map)
#target_size just putted for reference ,we have to change when actual coding
#It is a size of input image to resize

VALIDATION_DIR = "C:/Users/MONSTER VISION/Desktop/data/catdog/test_set/"
validation_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=2,
                                                              class_mode='categorical',
                                                              target_size=(374, 374))

#x,y=validation_generator.next()
label_map2 = (validation_generator.class_indices)
print(label_map2)

model=tf.keras.models.Sequential([      
        tf.keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',strides=1, input_shape=(374,374, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),  
        
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),  
        
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),
        
        #Added here Extra Layer to decrese output size to(6,6,512)
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu',padding='same',strides=1),
        tf.keras.layers.MaxPooling2D((3,3),strides=2,padding='same'),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation='relu'),
        tf.keras.layers.Dense(4096,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
        
        ])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history=model.fit_generator(train_generator,epochs=1,steps_per_epoch=2,verbose=1,
                            validation_data=validation_generator,validation_steps=2)


acc = history.history['acc']
val_acc = history.history['val_acc']


loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')


