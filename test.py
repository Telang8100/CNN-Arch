
import numpy as np
import tensorflow as tf
import time
import os
#from datetime import timedelta
from PIL import Image
#import glob
import scipy.misc #Version should be 1.1.0
#import configparser
#import pandas as pd
PATH='Path/To/Test/Data'
classes =['A','B',...]
model_file ="Path/To/pb/file"
path_test_images ="Path/to/A/class"
#path_test_images ="Path/to/B/class"
#path_test_images ="Path/to/C/class"



#thresholds =  np.array([0.1, 0.1, 0.4, 0.1, 0.1, 0.1])
thresholds = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])#How many classes that number of thresholds
[width, height] = [299, 299] #Image Size What is there while training
#s=.003921568
s = 0.00392157
m = 108.4112530
#m=55.71935604718378
#mean = 10.9564

conf_mat = np.zeros([len(classes), len(classes)], np.int32)
def load_model():
    with tf.io.gfile.GFile(model_file, "rb") as f:
        #File I/O wrappers without thread locking.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

session = tf.Session(graph=load_model())
for dir_, _, files in os.walk(path_test_images):
    relDir = os.path.relpath(dir_, path_test_images)
    print(relDir)
    no_test_imgs = (len(files))
    print(no_test_imgs)


    index=0
    X_test = np.zeros([no_test_imgs, height, width, 1], np.float32)
    #X_test = (X_test - mean) * s
    print(X_test.shape)

    Y_test = np.zeros(no_test_imgs, np.float32)
    tp = 0
    tf = 0
    time_diff = 0
    for fileName in files:
        img_path = os.path.join(path_test_images, relDir, fileName)
        try:
            millis = int(round(time.time() * 1000))
            print(img_path)
            img = np.array(Image.open(img_path))
            img = scipy.misc.imresize(img, (height, width))
            img = img.reshape((1, height, width, -1))
            img = (img.astype(np.float32)) * s
            
#            image = tf.io.read_file(img_path)  
#            image = tf.image.decode_bmp(image, channels=0)#Decode a JPEG-encoded image to a uint8 tensor.
#                                                  #Returns:- A Tensor of type uint8.
#
#            img = tf.image.resize(image, [299,299])#Resize images to size using the specified method.
#                                              #Returns:-If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
#                                              #If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]
#            img /= 255.0 
            for i in range(len(classes)):
                if classes[i] in img_path:
                    label = i
                    break
            index = index + 1
            predict_prob = session.run("output/Softmax:0", {"input:0": img})
            #predict_prob = session.run("predictions/Softmax:0", {"input_2:0": img})

            print("predict_prob : ",predict_prob)
            pred = ''
            val = []
            defectcnt = 0
            for i, j in enumerate(predict_prob[0]):
                #print("i",i)
                #print("j",j)
                if j > thresholds[i]:
                    pred += classes[i] + ''
                    print("pred :",pred)
                    val.append(j)
                    defectcnt += 1
            #print("pred",pred)
            with open(PATH + 'miss_class train_84.txt', 'a') as fo:
                fo.write("%s ----> %s\n " % (img_path, pred))
            fo.close()
            #print(val)
            # print Y_test[step_test:step_test+1]
            if 'GoodTest' in pred:

                # no defects found or high enough probability threshold implies good image
                predict_label = 0

            elif len(val) > 1:
                # if len(val.T)>1:
                for k in range(0, len(classes)):
                    if predict_prob[0][k] == max(val):
                        predict_label = k
                        #if len(predict_label) == 0:
             #   predict_label = classes[predict_prob.argmax()]

            elif len(val)==1:
               predict_label = np.argwhere(predict_prob == val).tolist()[0][1]
            else:
                predict_label= predict_prob.argmax()
            print(predict_label)

            if predict_label == label:
                tp += 1
            else:
                tf += 1
            conf_mat[int(label)][predict_label] += 1
            print(conf_mat)
            curr_millis = int(round(time.time() * 1000))
            time_diff += (curr_millis - millis)
            time_d=(curr_millis - millis)
            print( time_d)
            accuracy  = ((tp * 100) / no_test_imgs)
            print("accuracy: %.4f" % accuracy)
        except:
            print("images error")
            with open(PATH+'corrupted_images.txt', 'a') as f:
                f.write(img_path)

    

