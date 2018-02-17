import tensorflow as tf, numpy as np, imageio, matplotlib.pyplot as plt, os, cv2, traceback,random


"""
Create the train and test data. Inputs are stored as paths to the images, and expected outputs as one-hot vectors.
"""
def create_data():
    dir_file = {'Marcel-Train/A':0, 'Marcel-Train/B':1, 'Marcel-Train/C':2, 'Marcel-Train/Five':3, 'Marcel-Train/Point':4,'Marcel-Train/V':5,'Marcel-Test/A':0, 'Marcel-Test/B':1, 'Marcel-Test/C':2, 'Marcel-Test/Five':3, 'Marcel-Test/Point':4,'Marcel-Test/V':5}
    labels = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
    class_size = [0,0,0,0,0,0]
    files = []
    train_data = []
    test_data = []
    for dirname, dirnames, filenames in os.walk('Marcel-Train/'):
        for filename in filenames:
            if filename.endswith('jpg'):
                    train_data.append([os.path.join(dirname,filename),labels[dir_file[dirname]]])
    for dirname, dirnames, filenames in os.walk('Marcel-Test/'):
        for filename in filenames:
            if filename.endswith('jpg'):
                test_data.append([os.path.join(dirname,filename),labels[dir_file[dirname]]])
    return train_data, test_data


"""
Convolutional Layer with Max Pooling and Local Response Normalization
"""
def conv_layer(in_layer,out_chan,size,sigma=0.01,b=0.01,strd=[1,1,1,1],pool=True):
    in_chan = in_layer.shape.as_list()[3]
    w = tf.Variable(tf.truncated_normal([size,size,in_chan,out_chan],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_chan]))
    h = tf.nn.relu(tf.nn.conv2d(in_layer, w, strides=strd,padding='VALID')+b)
    p = tf.nn.max_pool(h,ksize = [1,3,3,1], strides = [1,2,2,1], padding='VALID')
    n = tf.nn.local_response_normalization(p, depth_radius=min(5,out_chan-2))
    n1 = tf.nn.local_response_normalization(h,depth_radius=min(5,out_chan-2))
    if pool:
        return w,b,h,p,n
    return w,b,h,n1


"""
Fully Connected Layer
"""
def conn_layer(in_layer,out_nodes,op_layer=False,sigma=0.1,b=0.0):
    i_s = in_layer.shape.as_list()
    print(i_s)
    in_layer2 = in_layer
    if len(i_s) > 2:
        in_layer2 = tf.reshape(in_layer,[-1,i_s[1]*i_s[2]*i_s[3]])
        w = tf.Variable(tf.truncated_normal([i_s[1]*i_s[2]*i_s[3],out_nodes],stddev=sigma))
    else:
        w = tf.Variable(tf.truncated_normal([i_s[-1],out_nodes],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_nodes]))
    h = tf.matmul(in_layer2,w)+b
    if not op_layer:
        h = tf.nn.relu(h)
    return w,b,h


"""
The architecture: 3 conv layers and  2 fc layers with dropout
"""
x = tf.placeholder(tf.float32, shape=[None,76*66*3])
y = tf.placeholder(tf.float32, shape=[None,6])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_img = tf.reshape(x,[-1,76,66,3])
w1,b1,h1,p1,n1 = conv_layer(x_img,9,5)
w2,b2,h2,p2,n2 = conv_layer(n1,27,3)
w3,b3,h3,p3,n3 = conv_layer(n2,51,3)
w4,b4,h4 = conn_layer(n3,1024)
h4_drop = tf.nn.dropout(h4,keep_prob)
w5,b5,h5 = conn_layer(h4_drop,512)
h5_drop = tf.nn.dropout(h5,keep_prob)
w6,b6,y_ = conn_layer(h5_drop,6,op_layer=True)


"""
Loss function: Softmax Cross Entropy
"""
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

"""
Adaptive moments for training
"""
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


"""
Compare predicted classes vs actual classes
"""
correct_prediction = tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32)

"""
Saver object to save and restore variables
"""
saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'w6':w6,'b6':b6})


"""
Get training data in a sequential manner
"""
def get_batch(start,end,files):
    train_array = [cv2.resize(imageio.imread(file)[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3) for file,lab in files[start:end]]
    train_labels = [lab for file,lab in files[start:end]]
    #print(train_array)
    #print(train_labels)
    return [train_array, train_labels]


"""
Get training data randomly
"""
def get_batch_random(files, batch_sz, f_read_dict):
    train_array = []
    train_labels = []
    for i in range(batch_sz):
        j = random.randint(0,2609)
        while j in f_read_dict:
            j = random.randint(0,2609)
        train_array.append(cv2.resize(imageio.imread(files[j][0])[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3))    
        train_labels.append(files[j][1])
        if len(f_read_dict) == 2609:
            f_read_dict = {}
    return [train_array, train_labels]

"""
Visualize output of a convolutional layer
"""
def visualize_layer(layer,sess,feed):
    unit = sess.run(layer,feed_dict = feed)
##    m = unit[0][0][0][0]
##    for i in range(unit.shape[0]):
##        for j in range(unit.shape[1]):
##            for k in range(unit.shape[2]):
##                for l in range(unit.shape[3]):
##                    m = max(m,unit[i][j][k][l])
##    unit = unit*255/m
    cv2.imshow('frame',unit[0,:,:,:3])
    cv2.waitKey(1)


"""
check validation accuracy
"""
def validate(test_data):
    acc = 0
    ls2 = 0
    try:
        for file, lab in test_data:
            ip = cv2.resize(imageio.imread(file)[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3)
            acc += correct_prediction.eval(feed_dict={x:[ip],y:[lab],keep_prob:1.0})
            ls2 += loss.eval(feed_dict={x:[ip], y:[lab], keep_prob:1.0})
        acc /= len(test_data)
        ls2 /= len(test_data)
        return acc,ls2
    except:
        traceback.print_exc()

"""
Train the model. Inputs: number of epochs, learning rate, train and test data, and whether to continue training model or start afresh
"""
def train(epochs,batch_sz,epsilon,train_data,test_data,reload=False):
    ls = []
    ls2 = []
    acc = []
    files_read = {}
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        if reload:
            try:
                saver.restore(sess, 'model6.ckpt')
            except:
                print("Error restoring checkpoint; starting afresh")
                
        for e in range(epochs):
            print(e+1)
            for b in range(0,2600-batch_sz,batch_sz):
                ip = get_batch_random(train_data, batch_sz,files_read)
                train_step.run(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:0.5})
                ls.append(loss.eval(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:1.0}))
                #print(sess.run(y_,feed_dict={x:ip[0]}))
                #visualize_layer(p2,sess,{x:[view_img]})
            if (e%10 == 0) and e>0:
                a,l = validate(test_data)
                acc.append(a)
                ls2.append(l)
        save_path = saver.save(sess, "model6.ckpt")
        print("Model saved at ", save_path)
        x1 = [i for i in range(len(ls))]
        x2 = [i for i in range(len(acc))]
        x3 = [i for i in range(len(ls2))]
        plt.figure(1)
        plt.plot(x1,ls)
        plt.figure(2)
        plt.plot(x2,acc)
        plt.figure(3)
        plt.plot(x3,ls2)
        plt.show()

"""
Test the model without training.
"""
def test():
    dir_file = {'Marcel-Train/A':0, 'Marcel-Train/B':1, 'Marcel-Train/C':2, 'Marcel-Train/Five':3, 'Marcel-Train/Point':4,'Marcel-Train/V':5,'Marcel-Test/A':0, 'Marcel-Test/B':1, 'Marcel-Test/C':2, 'Marcel-Test/Five':3, 'Marcel-Test/Point':4,'Marcel-Test/V':5}
    labels = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
    test_data = []
    for dirname, dirnames, filenames in os.walk('Marcel-Test/'):
        for filename in filenames:
            if filename.endswith('jpg'):
                test_data.append([os.path.join(dirname,filename),labels[dir_file[dirname]]])
    with tf.Session() as sess:
        saver.restore(sess, 'model6.ckpt')
        acc = 0
        for file, lab in test_data:
            img = cv2.resize(imageio.imread(file)[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3)
            cv2.imshow('frame',sess.run(p1,feed_dict={x:[img]})[0,:,:,:3])
            cv2.waitKey(1)
            acc += correct_prediction.eval(feed_dict={x:[img], y:[lab],keep_prob:1.0})
        acc/=len(test_data)
    print(acc)

"""
Set up the data and start training or testing.
"""
train_data,test_data = create_data()
print('train data imgs: ',len(train_data))
print('test data imgs: ',len(test_data))
view_img = cv2.resize(imageio.imread('Marcel-Test/A/A-uniform01.jpg')[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3)
train(200,10,1e-4,train_data,test_data,reload=True)
#test()


                          
                
    
