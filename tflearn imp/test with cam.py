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
convnet = regression(convnet, optimizer='sgd', learning_rate=0.00001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.load('NewModels/m1.model')

fgbg = cv2.createBackgroundSubtractorMOG2()
n='None'
count=0
cap = cv2.VideoCapture(0)
preds = glob('NewData/Train/*')

while(1):
	ret, frame = cap.read()
	if ret == True:
		fgmask = fgbg.apply(frame)
		
		cv2.imshow('frame',frame[100:450,80:330,:])
		cv2.imshow('fgmask',fgmask[100:450,80:330])


		fgmask = fgmask[100:450,80:330]
		fgmask = cv2.resize(fgmask ,(128,128), interpolation = cv2.INTER_CUBIC)
		fgmask = np.reshape(fgmask , [1,128,128, 1])
		print('Predicted :',preds[np.argmax(model.predict(fgmask))])	

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	else:
		break


cap.release()
cv2.destroyAllWindows()
