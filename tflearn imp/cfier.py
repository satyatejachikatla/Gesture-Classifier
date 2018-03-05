import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import numpy as np 
from sklearn.utils import shuffle

dataset_path='../Data/Train'
X , Y = image_preloader(dataset_path, image_shape=(128, 128),   mode='folder', categorical_labels=True,   normalize=False, grayscale=True)
dataset_path='../Data/Test'
test_x , test_y = image_preloader(dataset_path, image_shape=(128, 128),   mode='folder', categorical_labels=True,   normalize=False, grayscale=True)

X,Y=shuffle(X,Y,random_state=0)
test_x,test_y=shuffle(test_x,test_y,random_state=0)

X=np.array(X)
test_x=np.array(test_x)

X=X.reshape([-1,128,128,1])
test_x=test_x.reshape([-1,128,128,1])

convnet = input_data(shape=[None, 128, 128, 1], name='input')

convnet = conv_2d(convnet, 128, 8, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 4, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = fully_connected(convnet, 64, activation='relu')	
convnet = dropout(convnet, 0.8)
	
convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.load('Models/m3.model')

model.fit({'input': X}, {'targets': Y}, n_epoch=7, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=128, show_metric=True, run_id='test',batch_size=1)

model.save('Models/m3.1.model')
