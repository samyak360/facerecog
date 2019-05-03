#!/usr/bin/python3

import cv2
import numpy as np
#loading face xml database
face_cascade = cv2.CascadeClassifier('face.xml')
#starting webcam
cap = cv2.VideoCapture(0)

d=0
basic_location="/home/samyak/Desktop/face_recognition/trainingdata/s5/"

while(d<10):
	#reading camera frame
	status,img=cap.read()
	#converting color image into grayscale image
	grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#trying multi angle in grayscaled img       #tunning parameter
	faces=face_cascade.detectMultiScale(grayimg,1.12,5)
	#applying iteration in multi scaled variable
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		#gray face image data
		roi_gray=grayimg[y:y+h,x:x+w]
		roi_color=img[y:y+h,x:x+w]
		if cv2.waitKey(5) & 0xFF == ord('c'):
			cv2.imwrite(basic_location+str(d)+".jpg",roi_color)
			d=d+1
	cv2.imshow('img',img)
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()


	
