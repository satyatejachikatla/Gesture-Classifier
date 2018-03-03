import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

#HSV - Hue Sat Value
while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_skin = np.array([10,int(0.1*255),0])
	upper_skin = np.array([180,int(0.4*255),255])
	mask = cv2.inRange(hsv, lower_skin, upper_skin)
	res = cv2.bitwise_and(frame, frame, mask = mask)
	cv2.imshow('res',res)

	'''
	Smooth bluer
	ksize = 2
	kernal = np.ones((ksize,ksize),np.float32)/(ksize*ksize)
	smoothed = cv2.filter2D(res, -1, kernal)
	cv2.imshow('smoothed',smoothed)
	'''

	blur = cv2.GaussianBlur(res,(15,15),0)
	cv2.imshow('gblur',blur)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()	
