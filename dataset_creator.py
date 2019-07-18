# -*- coding: utf-8 -*-

import os 
import cv2
import numpy as np

dataset = 'Full images'
target = 'Dataset'

faceCascade = cv2.CascadeClassifier("Assets/haarcascade_frontalface_default.xml")

for persons in (os.listdir(dataset)):
        for person in (os.listdir(str(dataset)+"/"+str(persons))):
            
            gray = cv2.imread(str(dataset)+"/"+str(persons)+"/"+str(person), 1)
            frame = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(
                frame, 
                scaleFactor = 1.3,
                minNeighbors = 5,
                minSize = (30, 30)
            )
            
            for (x,y,w,h) in faces:
                frame = gray[y-20:y+h+20, x-12:x+w+12]
                
            if os.path.exists(str(target)+"/"+str(persons)):
                cv2.imwrite(str(target)+"/"+str(persons)+"/"+str(person), frame)
            else:
                os.mkdir(os.path.join(target, persons))
                cv2.imwrite(str(target)+"/"+str(persons)+"/"+str(person), frame)

