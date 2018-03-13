import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import numpy as np 
from sklearn.utils import shuffle

dataset_path='./CacBlackData/Train'
X , Y = image_preloader(dataset_path, image_shape=(128 , 128),   mode='folder', categorical_labels=True,   normalize=False, grayscale=True)
dataset_path='./CacBlackData/Test'
test_x , test_y = image_preloader(dataset_path, image_shape=(128 , 128),   mode='folder', categorical_labels=True,   normalize=False, grayscale=True)

X=np.array(X)
test_x=np.array(test_x)
Y=np.array(Y)
test_y=np.array(test_y)


X=X.reshape([-1,128 , 128, 1])
test_x=test_x.reshape([-1,128 , 128, 1])

convnet = input_data(shape=[None, 128 , 128, 1], name='input')

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

model.load('CacBlackModels/m1.model')

print('Test Acc:',model.evaluate(test_x,test_y))
print('Train Acc:',model.evaluate(X,Y))

print('#--------For train data------------#')
a_p=[]
for i in range(len(X)):
	pred = np.argmax(model.predict([X[i]]))
	act = np.argmax(Y[i])
	a_p+=[(act,pred)]
	print('Actual',act,'Predicted',pred,'Matched',act==pred)

print('#--------For train data------------#')
a_p=[]
for i in range(len(test_x)):
	pred = np.argmax(model.predict([test_x[i]]))
	act = np.argmax(test_y[i])
	a_p+=[(act,pred)]
	print('Actual',act,'Predicted',pred,'Matched',act==pred)
