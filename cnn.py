import tensorflow as tf, numpy as np, imageio, matplotlib.pyplot as plt, cv2, traceback

"""
Convolutional Layer with Max Pooling and Local Response Normalization
"""
def conv_layer(in_layer,out_chan,size,sigma=0.01,b=0.0,strd=[1,1,1,1],pool=True):
    in_chan = in_layer.shape.as_list()[3]
    w = tf.Variable(tf.truncated_normal([size,size,in_chan,out_chan],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_chan]))
    h = tf.nn.relu(tf.nn.conv2d(in_layer, w, strides=strd,padding='VALID')+b)
    n = tf.nn.local_response_normalization(h, depth_radius=min(5,out_chan-2))
    p = tf.nn.max_pool(n,ksize = [1,3,3,1], strides = [1,2,2,1], padding='VALID')
    n1 = tf.nn.local_response_normalization(h,depth_radius=min(5,out_chan-2))
    if pool:
        return w,b,h,p
    return w,b,h,n1


"""
Fully Connected Layer
"""
def conn_layer(in_layer,out_nodes,op_layer=False,sigma=0.01,b=0.0):
    i_s = in_layer.shape.as_list()
    #print(i_s)
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
    r = tf.nn.l2_loss(w)
    return w,b,h,r


"""
The architecture: 3 conv layers and  2 fc layers with dropout
"""
x = tf.placeholder(tf.float32, shape=[None,128*128*1])
y = tf.placeholder(tf.float32, shape=[None,5])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_img = tf.reshape(x,[-1,128,128,1])
w1,b1,h1,p1 = conv_layer(x_img,96,3)
w2,b2,h2,p2 = conv_layer(p1,48,3)
w3,b3,h3,p3 = conv_layer(p2,24,3)
w4,b4,h4,r4 = conn_layer(p3,2048)
h4_drop = tf.nn.dropout(h4,keep_prob)
w5,b5,h5,r5 = conn_layer(h4_drop,1024)
h5_drop = tf.nn.dropout(h5,keep_prob)
w6,b6,y_,r6 = conn_layer(h5_drop,5,op_layer=True)


"""
Loss function: Softmax Cross Entropy
"""
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
reg = r4+r5+r6
loss = loss0 + 0.01*reg

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
Visualize output of a convolutional layer
"""
def visualize_layer(layer,sess):
    img = imageio.imread('./Data/Our-Train2/test/1/n_41.jpg')
    ch = 1
    if len(img.shape) > 2:
        ch = min(3,img.shape[2])
        img = img[:,:,:ch]
    ip = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA).reshape(128*128*ch)
    unit = sess.run(layer,feed_dict = {x:[ip]})
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
def validate(net_loader,sess):
    acc = 0
    ls2 = 0
    test_data  = net_loader.test_data
    try:
        for file, lab in test_data:
            #print(file, lab)
            ip = net_loader.get_single_img(file)
            #print('predicted: ',np.argmax(sess.run(y_,feed_dict={x:[ip],keep_prob:1.0})))
            #print('actual: ',np.argmax(lab), ' ',lab)
            acc += correct_prediction.eval(feed_dict={x:[ip],y:[lab],keep_prob:1.0})
            ls2 += loss.eval(feed_dict={x:[ip], y:[lab], keep_prob:1.0})
        acc /= len(test_data)
        ls2 /= len(test_data)
        print('loss: ',ls2)
        return acc,ls2
    except:
        traceback.print_exc()

"""
Train the model. Inputs: number of epochs, learning rate, train and test data, and whether to continue training model or start afresh
"""
def train(epochs,batch_sz,epsilon,net_loader,reload):
    print('epochs:',epochs,' learning rate:',epsilon,' batch size:', batch_sz,' reload:',reload)
    ls = []
    ls2 = []
    acc = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = 'model4.ckpt'
        if reload == 'True':
            try:
                saver.restore(sess, net_loader.model_dir+ckpt)
                print("Model reloaded successfully.")
            except tf.errors.NotFoundError:
                print("Model "+ckpt+" not found, will create new file")
        elif reload == 'False':
            print("'Reload' set to 'False', starting afresh")
                
        for e in range(epochs):
            print(e+1)
            for b in range(0,net_loader.train_size,batch_sz):
                ip = net_loader.get_batch_random(batch_sz)
                train_step.run(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:0.8})
                if b%2 == 0:
                    ls.append(loss.eval(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:1.0}))
                #print(sess.run(y_,feed_dict={x:ip[0]}))
                #visualize_layer(n3,sess)
            if ((e+1)%(epochs/10) == 0) or epochs <= 50:
                a,l = validate(net_loader,sess)
                acc.append(a)
                ls2.append(l)
                if len(acc)>1 and a>np.amax(acc):
                    save_path = saver.save(sess, net_loader.model_dir+ckpt+str(e+1))
                    print('Model saved at ', save_path)
        save_path = saver.save(sess, net_loader.model_dir+ckpt)
        print('Model saved at ', save_path)
        x1 = [i for i in range(len(ls))]
        x2 = [i for i in range(len(acc))]
        x3 = [i for i in range(len(ls2))]
        plt.figure('train loss')
        plt.plot(x1,ls)
        plt.figure('test acc')
        plt.plot(x2,acc)
        plt.figure('test loss')
        plt.plot(x3,ls2)
        plt.show()

"""
Test the model without training.
"""
def test(net_loader):
    with tf.Session() as sess:
        saver.restore(sess, net_loader.model_dir+ckpt)
        acc = 0
        for file, lab in net_loader.test_data:
            img = net_loader.get_single_img(file)
            #cv2.imshow('frame',sess.run(p1,feed_dict={x:[img]})[0,:,:,:3])
            #cv2.waitKey(1)
            acc += correct_prediction.eval(feed_dict={x:[img], y:[lab],keep_prob:1.0})
        acc/=net_loader.test_size
    print(acc)
