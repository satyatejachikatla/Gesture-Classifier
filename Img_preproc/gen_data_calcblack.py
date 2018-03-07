import numpy as np
import cv2
#from time import gmtime, strftime
from random import randint
from glob import glob

n=input('Enter Class Folder:').strip()
count = len(glob('../Data/'+n+'/*'))
print('Currently',count,'images present in folder.')
cap = cv2.VideoCapture(0)

#---------------------------------------------------------------------------------------#
#Capture Background
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
#Actual capture of images
print('Started the cam to record')
cap_flag = False
while True:
	ret, frame = cap.read()
	if ret == True:

		target = frame
		hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		thresh = cv2.merge((thresh,thresh,thresh))
		#res = cv2.bitwise_and(target,thresh)
		thresh = cv2.bitwise_not(thresh)
		#cv2.imshow("res", res)

		cv2.imshow('thresh', thresh[:256,:256])	

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == 32:
			cap_flag = not cap_flag
		if cap_flag:
			count+=1	
			img_name = 'Data/'+n+'/'+str(count)+'.jpg'
			cv2.imwrite(img_name,thresh[:256,:256])
			print('Saved : ',img_name)
	else:
		break		

cap.release()
cv2.destroyAllWindows()
