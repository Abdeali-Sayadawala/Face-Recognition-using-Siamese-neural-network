# -*- coding: utf-8 -*-

import os 
import cv2
import numpy as np
from reconet import Reconet

dim = 200

x1 = np.load('training data/x1.npy')
x2 = np.load('training data/x2.npy')
y = np.load('training data/y.npy')

"""x1 = x1.reshape((x1.shape[0], dim**2*3)).astype(np.float32)
x2 = x2.reshape((x2.shape[0], dim**2*3)).astype(np.float32)"""

rec = Reconet()

parameters = {
        'batch_size' : 2,
        'epochs' : 7,
        'callbacks' : None,
        'val_data' : None
        }

rec.fit([x1, x2], y, hyperparameters=parameters)
rec.save_model('model.h5')


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
    



"""name = "Shraddha"
for i in c:
    i = i.reshape((1, 200, 200, 3))
    print(str(name)+" vs Shraddha")
    for j in prin2:
        j = j.reshape((1, 200, 200, 3))
        pr = rec.predict([i, j])
        print(pr)
    print(str(name)+" vs Keanu")
    for j in prin1:
        j = j.reshape((1, 200, 200, 3))
        pr = rec.predict([i, j])
        print(pr)
    name = "Keanu"""
