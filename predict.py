# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from keras.models import load_model
from reconet import Reconet
import keras.backend as K



def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
        

rec = Reconet()
model = load_model("model.h5", custom_objects={'contrastive_loss':contrastive_loss})


namelabels = list()

faceCascade = cv2.CascadeClassifier("Assets/haarcascade_frontalface_default.xml")
vc = cv2.VideoCapture(0)

while True:
    ret, frameo = vc.read()
    gray = cv2.cvtColor(frameo, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,        
        minSize=(30,30)
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frameo, (x-12,y-20), (x+w+12,y+h+20), (0,0,255), 1)
        frame = gray[y-20:y+h+20, x-12:x+w+12]
        
    cv2.imshow("detect", frameo)
        
    cv2.imwrite("Custom/1.jpg", frame)
    c = rec.prepare_images('Custom')
    c = c[0].reshape((1, 200, 200, 3))
    all_pred = list()
    
    for person in os.listdir("Dataset"):
        per = list()
        namelabels.append(person)
        pr = rec.prepare_images("Dataset/"+str(person))
        
        for i in pr:
            i = i.reshape((1, 200, 200, 3))
            pred = rec.predict([c, i])
            per.append(pred)
        
        mn = np.mean(per)
        all_pred.append(mn)
    ind = all_pred.index(min(all_pred))
    print(namelabels[ind])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()