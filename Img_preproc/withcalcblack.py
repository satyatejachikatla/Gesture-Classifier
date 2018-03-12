import cv2
import numpy as np


cap = cv2.VideoCapture(0)

#---------------------------------------------------------------------------------------#
#Capture background and understand its rang
print('Enter \'c\' to capture empty background')
while True:
	ret, frame = cap.read()


	roi = frame[:256,:256,:]
	hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
	target = frame
	hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

	cv2.rectangle(frame,(0,0),(256,256),(0,255,0),3)
	cv2.imshow('frame',frame)

	if ret == True:
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('c'):
			# calculating object histogram
			roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
			# normalize histogram and apply backprojection
			cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
			break

	else:
		break
cv2.destroyAllWindows()
#---------------------------------------------------------------------------------------#
#without background mask
while True:
	ret, frame = cap.read()
	if ret == True:
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	else:
		break

	target = frame
	hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	thresh = cv2.merge((thresh,thresh,thresh))
	res = cv2.bitwise_and(target,thresh)
	thresh = cv2.bitwise_not(thresh)
	#cv2.imshow("res", res)

	cv2.imshow('thresh', thresh)
cap.release()
cv2.destroyAllWindows()
#---------------------------------------------------------------------------------------#