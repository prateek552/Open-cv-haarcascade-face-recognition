# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:19:24 2018

@author: Heller
"""

import numpy as np
import cv2
cam=cv2.VideoCapture(0)
fc=cv2.CascadeClassifier('intel.xml')
eye=cv2.CascadeClassifier('eye.xml')
while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        gray1=gray[y:y+h,x:x+h]
        color1=frame[y:y+h,x:x+h]
        eyes=eye.detectMultiScale(gray1)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(color1,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()