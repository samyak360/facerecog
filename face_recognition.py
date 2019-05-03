#!/usr/bin/python 

from selenium import webdriver 
from time import sleep 
from getpass import getpass 
import os
import cv2
import numpy as np
#loading face xml database
face_cascade = cv2.CascadeClassifier('face.xml')

subjects=["","samyak","palak","shivansh","harsh","ashish"]
labels=[]
faces=[]
basiclocation="/home/samyak/Desktop/face_recognition/trainingdata/"
dirs = sorted(os.listdir(basiclocation))

for dir_name in dirs:
	#our subject directories start with letter 's' so
	#ignore any non-relevant directories if any
	if not dir_name.startswith("s"):
		continue
	for img_name in sorted(os.listdir(basiclocation+dir_name)):
		labels.append(int(dir_name.replace("s","")))
		image=cv2.imread(basiclocation+dir_name+"/"+img_name)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces.append(gray)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

d=0
cap = cv2.VideoCapture(0)
while(d<1):
	#reading camera frame
	status,img=cap.read()
	#converting color image into grayscale image
	grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#trying multi angle in grayscaled img       #tunning parameter
	facedetect=face_cascade.detectMultiScale(grayimg,1.12,5)
	#applying iteration in multi scaled variable
	for (x,y,w,h) in facedetect:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		#gray face image data
		roi_gray=grayimg[y:y+h,x:x+w]
		roi_color=img[y:y+h,x:x+w]
		if cv2.waitKey(5) & 0xFF == ord('c'):
			resultcode = face_recognizer.predict(roi_gray)
			resulttext = subjects[resultcode[0]]
			d=d+1
			print(resulttext)
	cv2.imshow('img',img)
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


# automatic facebook login 

if resulttext=="samyak" :
	usr='' 
	pwd = '' 

if resulttext=="palak" :
	usr='' 
	pwd = '' 

if resulttext=="shivansh" :
	usr='' 
	pwd = '' 

if resulttext=="harsh" :
	usr='' 
	pwd = '' 

if resulttext=="ashish" :
	usr='' 
	pwd = '' 

driver = webdriver.Firefox() 
driver.get('https://www.facebook.com/') 
print ("Opened facebook") 
sleep(1) 

username_box = driver.find_element_by_id('email') 
username_box.send_keys(usr) 
print ("Email Id entered") 
sleep(1) 

password_box = driver.find_element_by_id('pass') 
password_box.send_keys(pwd) 
print ("Password entered") 

login_box = driver.find_element_by_id('loginbutton') 
login_box.click() 

print ("Done") 
input('Press enter to quit') 
driver.quit() 
print("Finished") 

