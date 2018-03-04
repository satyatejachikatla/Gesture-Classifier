import numpy as np
import cv2
from time import gmtime, strftime
from random import randint
from glob import glob

fgbg = cv2.createBackgroundSubtractorMOG2()
n='5'
count = len(glob('Data/'+n+'/*'))
cap = cv2.VideoCapture(0)

while(1):
	ret, frame = cap.read()
	if ret == True:
		fgmask = fgbg.apply(frame)
		
		
		cv2.imshow('frame',frame[100:450,80:330,:])
		cv2.imshow('fgmask',fgmask[100:450,80:330])
		
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == 32:
			count+=1	
			img_name='Data/'+n+'/'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'.jpg'
			cv2.imwrite(img_name,cv2.cvtColor(fgmask[100:450,80:330],cv2.COLOR_GRAY2RGB))
			print('Saved : ',img_name , count)
	else:
		break

cap.release()
cv2.destroyAllWindows()
