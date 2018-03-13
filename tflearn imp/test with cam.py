import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import numpy as np 
from sklearn.utils import shuffle
import cv2
from glob import glob

convnet = input_data(shape=[None, 128,128, 1], name='input')

convnet = conv_2d(convnet, 128, 8, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 4, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = fully_connected(convnet, 64, activation='relu')	
convnet = dropout(convnet, 0.8)
	
convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='sgd', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.load('CacBlackModels/m1.model')

cap = cv2.VideoCapture(0)
preds = glob('CacBlackData/Train/*')

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
	cv2.imshow('frame',frame[:256,:256,:])

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
cv2.imshow('Actual',frame[:1,:1])
cv2.imshow('Output',frame[:1,:1])
cv2.moveWindow('Actual', 100,100)
cv2.moveWindow('Output', 600,100)
#---------------------------------------------------------------------------------------#
#Actual capture of images
print('Started the cam to predict')
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
		thresh = cv2.bitwise_not(thresh)
		op = frame[:256,:256,:]
		cv2.imshow('Actual',op)
		
		t_=thresh = thresh[:256,:256,:]
		thresh = cv2.resize(thresh ,(128,128), interpolation = cv2.INTER_CUBIC)
		thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
		thresh = np.reshape(thresh , [1,128,128, 1])
		ans = preds[np.argmax(model.predict(thresh))]
		print('Predicted :',ans)	
		
		op_ = np.zeros((100,256,3))
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(op_,ans.strip('CacBlackData/Train/'),(0,90), font, 2,(255,0,0),2,cv2.LINE_AA)
		op = np.vstack((t_,op_))
		cv2.imshow('Output', op)
		

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	else:
		break		

cap.release()
cv2.destroyAllWindows()
