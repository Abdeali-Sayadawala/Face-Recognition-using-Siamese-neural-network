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
        
persons = list()
namelabels = list()
for person in os.listdir("Dataset"):
    namelabels.append(person)
    pr = rec.prepare_images("Dataset/"+str(person))
    pr = pr[0].reshape((1, 200, 200, 3))
    persons.append(pr)



model = load_model("model45.h5", custom_objects={'contrastive_loss':contrastive_loss})



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
        frame = frameo[y-20:y+h+20, x-12:x+w+12]
        cv2.rectangle(frameo, (x-12,y-20), (x+w+12,y+h+20), (0,0,255), 1)
        
    cv2.imshow("detect", frameo)
    
    frame = cv2.resize(frame, (200, 200))
    frame = np.array(np.reshape(frame, (200, 200, 3)))/255
    c = frame.reshape((1, 200, 200, 3))
        
    all_pred = list()
        
    for per in persons:
        pred = rec.predict([c, per])
        all_pred.append(pred)
        
    ind = all_pred.index(min(all_pred))
    print(namelabels[ind])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()