# -*- coding: utf-8 -*-

import cv2
import os
import time
import numpy as np

i = 1
frames = list()
name = input("Enter Name: ")

faceCascade = cv2.CascadeClassifier("Assets/haarcascade_frontalface_default.xml")

vc = cv2.VideoCapture(0)
for i in range(0, 1001, 5):
    ret, frame = vc.read()
         
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x,y,w,h) in faces:
        if i%100 == 0 and i != 0:
            cv2.rectangle(frame, (x-12,y-20), (x+w+12,y+h+20), (0,0,255), 2)
            frames.append(frame[y-20:y+h+20, x-12:x+w+12])
        else:
           cv2.rectangle(frame, (x-12,y-20), (x+w+12,y+h+20), (0,255,0), 2) 
        #frame = frame[y:y+h, x:x+w]
        
    frame = cv2.flip(frame, 1)
    
    print(i)
        
    cv2.imshow('frame', frame)
    if i%100 == 0 and i != 0:
        time.sleep(1)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   

vc.release()
cv2.destroyAllWindows()


an = input("Save ?(y/n): ")

if an == "y":
    for f in range(len(frames)):
        if os.path.exists(name):
            cv2.imwrite("Dataset/"+str(name)+"/"+str(f+1)+".jpg", frames[f])
        else:
            os.mkdir(name)
            cv2.imwrite("Dataset/"+str(name)+"/"+str(f+1)+".jpg", frames[f])

    
