# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

dim = 200

dataset = 'Dataset'
target = 'training data'

images = list()

if len(os.listdir(dataset)) != 0:
    for person in os.listdir(dataset):
        if len(os.listdir(os.path.join(dataset, person))) != 0:
            image = list()
            for img in os.listdir(os.path.join(dataset, person)):
                img = cv2.imread(str(dataset)+"/"+str(person)+"/"+str(img))
                img = cv2.resize(img, (dim, dim))
                img = np.array(np.reshape(img, (dim,dim,3)))/255
                image.append(img)
            images.append(image)
        else:
            print("There are no images for "+str(person))
else:
    print("There is no data in Dataset. \n Please run the prepare dataset file after keeping person images in Full images")
    
    
sample1 = list()
sample2 = list()
label = list()

for i in range(len(images)):
    for j in range(len(images)):
        if i == j:
            for x in range(5):
                for y in range(5):
                    sample1.append(images[i][x])
                    sample2.append(images[j][y])
                    label.append(1)
        else:
            for x in range(5):
                for y in range(5):
                    sample1.append(images[i][x])
                    sample2.append(images[j][y])
                    label.append(0)
        
x1 = np.array(sample1)
x2 = np.array(sample2)
y = np.array(label)

np.save(str(target)+"/x1.npy", x1)
np.save(str(target)+"/x2.npy", x2)
np.save(str(target)+"/y.npy", y)
    