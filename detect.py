# -*- coding: utf-8 -*-
import numpy as np
import cv2

face_cas=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cas=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap=cv2.VideoCapture(0)

while True:
    #To read the video image frame by frame
    ret,img=cap.read() 
    # To  convert the image into gray scale the below code is used
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces=face_cas.detectMultiScale(gray, 1.3, 5)
    
    for (l,b,w,h) in faces:
        #To get face boundary
        cv2.rectangle(img, (l,b), (l+w, b+h), (255,0,255), 5)
        #To find region of image
        reg_img_gray= gray[b:int((b+h)), l:l+w]
        
        reg_img_color= img[b:int((b+h)), l:l+w]
        #To find eyes in the given region of image
        eye=eye_cas.detectMultiScale(reg_img_gray)
        im_it="swaroop.png"
        cv2.imwrite(im_it,reg_img_gray)
        
        for (el,eb,ew,eh) in eye:
            #To indicate eyes in the given roi
            cv2.rectangle(reg_img_color, (el,eb), (el+ew,eb+eh), (0,255,0), 2)
    
    #To show the image
    cv2.imshow('Face Detect',img)
    
    n=cv2.waitKey(30) & 0xff
    
    if n==27:
        break

cap.release()
cv2.destroyAllWindows()

