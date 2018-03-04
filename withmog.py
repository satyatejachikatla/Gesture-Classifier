import numpy as np
import cv2

#cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#history=10
cap = cv2.VideoCapture(0)

while(1):
	ret, frame = cap.read()
	if ret == True:
		fgmask = fgbg.apply(frame)
		#fgmask = fgbg.apply(frame,learningRate=1.0/history)
		#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		
		cv2.imshow('fgmask',frame)
		cv2.imshow('frame',fgmask)
		
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == 32:
			cv2.imwrite('temp.jpg',cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB))
	else:
		break
    

cap.release()
cv2.destroyAllWindows()
