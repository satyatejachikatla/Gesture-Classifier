import tensorflow as tf, numpy as np, imageio, matplotlib.pyplot as plt, cv2, traceback, os

"""
Convolutional Layer with Max Pooling and Local Response Normalization
"""
def conv_layer(in_layer,out_chan,size,sigma=0.01,b=0.0,strd=[1,1,1,1],pool=True):
    in_chan = in_layer.shape.as_list()[3]
    w = tf.Variable(tf.truncated_normal([size,size,in_chan,out_chan],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_chan]))
    h_ = tf.nn.conv2d(in_layer, w, strides=strd,padding='VALID')+b
    p = tf.nn.max_pool(h_,ksize = [1,4,4,1], strides = [1,2,2,1], padding='VALID')
    h = tf.nn.relu(p)
    n = tf.nn.local_response_normalization(h, depth_radius=max(0,min(4,out_chan-2)))
    if pool:
        return w,b,h,n
    h = tf.nn.relu(h_)
    n1 = tf.nn.local_response_normalization(h,depth_radius=max(min(4,out_chan-2)))
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
#double check layer inputs
output_classes = 6
x = tf.placeholder(tf.float32, shape=[None,128*128*1])
y = tf.placeholder(tf.float32, shape=[None,output_classes])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_img = tf.reshape(x,[-1,128,128,1])
w1,b1,h1,n1 = conv_layer(x_img,8,8)
w2,b2,h2,n2 = conv_layer(n1,4,8)
w3,b3,h3,n3 = conv_layer(n2,16,16)
w4,b4,h4,r4 = conn_layer(n2,2048)
h4_drop = tf.nn.dropout(h4,keep_prob)
w5,b5,h5,r5 = conn_layer(h4_drop,1024)
h5_drop = tf.nn.dropout(h5,keep_prob)
w6,b6,h6,r6 = conn_layer(h5_drop,512)
h6_drop = tf.nn.dropout(h6,keep_prob)
w7,b7,y_,r7 = conn_layer(h4_drop,output_classes,op_layer=True)


"""
Loss function: Softmax Cross Entropy
"""
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
reg = r4+r7
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
saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'w6':w6,'b6':b6,'w7':w7,'b7':b7})

"""
Visualize output of a convolutional layer
"""
def visualize_layer(layer,sess):
    img = cv2.imread('../N_Data/Train/1/1.jpg',0)
    ch = 1
    if len(img.shape) > 2:
        ch = min(3,img.shape[2])
        img = img[:,:,:ch]
    ip = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA).reshape(128*128*ch)
    unit = sess.run(layer,feed_dict = {x:[ip]})
    m = unit[0][0][0][0]
    for i in range(unit.shape[0]):
        for j in range(unit.shape[1]):
          for k in range(unit.shape[2]):
            for l in range(unit.shape[3]):
              m = max(m,unit[i][j][k][l])
    unit = unit*255/m
    cv2.imshow('frame',unit[0,:,:,:3])
    cv2.waitKey(1)


"""
check validation accuracy
"""
def validate(net_loader,sess,test=False):
    acc = 0
    ls2 = 0
    acc_t = 0
    ls_t = 0
    test_data  = net_loader.test_data
    step = 1
    out_str = 'test loss:'
    out_str2 = 'test acc:'

    if test == False:
        step = 4
        out_str = 'validation loss:'
        out_str2 = 'validation acc:'        
    try:
        for i in range(0,len(test_data),step):
            #print(file, lab)
            ip = net_loader.get_single_img(test_data[i][0])
            lab = test_data[i][1]
            #print('predicted: ',np.argmax(sess.run(y_,feed_dict={x:[ip],keep_prob:1.0})))
            #print('actual: ',np.argmax(lab), ' ',lab)
            acc += correct_prediction.eval(feed_dict={x:[ip],y:[lab],keep_prob:1.0})
            ls2 += loss.eval(feed_dict={x:[ip], y:[lab], keep_prob:1.0})
        acc /= len(test_data)/step
        ls2 /= len(test_data)/step
        print(out_str,ls2, out_str2,acc)
        return acc,ls2
    except:
        traceback.print_exc()


ckpt = 'model3.ckpt'
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

        acc_file = []
        prev_acc = -1
        prev_ls = 999999999
        if reload == True:
            try:
                saver.restore(sess, net_loader.model_dir+ckpt)
                print("Model reloaded successfully.")
                try:
                    acc_file = open(net_loader.model_dir+'prev_acc.txt','r')
                    prev_acc = np.float32(acc_file.readline().strip())
                    prev_ls = np.float32(acc_file.readline().strip())
                    acc_file.close()
                    print('previous test loss: ',prev_ls)
                    print('previous test accuracy: ',prev_acc)
                except OSError:
                    pass
            except tf.errors.NotFoundError:
                print("Model "+ckpt+" not found, will create new file")
        else:
            print("'Reload' set to 'False', starting afresh")

        for e in range(epochs):
            print(e+1)
            l = 0
            a = 0
            for b in range(0,net_loader.train_size,batch_sz):
                ip = net_loader.get_batch_random(batch_sz)
                train_step.run(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:0.5})
                l += loss.eval(feed_dict={x:ip[0],y:ip[1],keep_prob:1.0})
                a += np.mean(correct_prediction.eval(feed_dict={x:ip[0],y:ip[1],keep_prob:1.0}))
            l /= net_loader.train_size/batch_sz
            a /= net_loader.train_size/batch_sz
            print("Train loss: ",l)
            print("Train acc: ",a)
            ls.append(l)
            if ((e+1)%(epochs/10) == 0) or epochs <= 50:
                a,l = validate(net_loader,sess,True)
                if len(acc)<=1:
                    if a>=prev_acc:
                        save_path = saver.save(sess, net_loader.model_dir+ckpt)
                        print('Model saved at ', save_path)       
                elif a>=np.amax(acc) and a>=prev_acc:
                    save_path = saver.save(sess, net_loader.model_dir+ckpt)
                    print('Model saved at ', save_path)
                    acc_file = open(net_loader.model_dir+'prev_acc.txt','w')
                    acc_file.write(str(a[0])+'\n')
                    acc_file.write(str(l)+'\n')
                    acc_file.close()
                acc.append(a)
                ls2.append(l)
        a,l = validate(net_loader,sess,True)
        print("Final test loss:",l," ; Final test accuracy:",a)
##    save_path = saver.save(sess, net_loader.model_dir+ckpt)
##    print('Model saved at ', save_path)
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

"""
With video check
"""
def foo(net_loader, mirror=False):
        with tf.Session() as sess:
                saver.restore(sess, net_loader.model_dir+ckpt)
                cap = cv2.VideoCapture(0)
                
                while(True):
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    # Our operations on the frame come here
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray=gray[0:128*2,0:128*2]
                    if mirror == True:
                        gray = cv2.flip(gray,1)
                    ggray=gray
                    # Display the resulting frame
                    
                    cv2.imshow('gray',gray)
                    cv2.waitKey(1)
                    height, width = gray.shape[:2]
                    gray = cv2.resize(gray,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)
                    gray=np.reshape(gray,[1,128*128])
                    nn_img = sess.run(n2,feed_dict={x:gray,keep_prob:1.0})
                    for i in range(0,nn_img.shape[3]-3,3):
                        cv2.imshow('frame'+str(i),nn_img[0,:,:,i:i+3])
                        if cv2.waitKey(1) & 0xFF == ord(' '):
                            break
                    print(net_loader.nums_class[sess.run(tf.argmax(y_,1),feed_dict={x:gray,keep_prob:1.0})[0]])
                # When everything done, release the capture
                cap.release()
                cv2.destroyAllWindows()
def foo1(net_loader):
        with tf.Session() as sess:
                saver.restore(sess, net_loader.model_dir+ckpt)
                cap = cv2.VideoCapture(0)
                fgbg = cv2.createBackgroundSubtractorMOG2()
                while(1):
                    ret, frame = cap.read()
                    if ret == True:
                        fgmask = fgbg.apply(frame)
                        
                        cv2.imshow('frame',frame[100:450,80:330,:])
                        cv2.imshow('fgmask',fgmask[100:450,80:330])


                        fgmask = fgmask[100:450,80:330]
                        fgmask = cv2.resize(fgmask ,(128,128), interpolation = cv2.INTER_CUBIC)
                        fgmask = np.reshape(fgmask , [1,128*128*1])
                        print('Predicted :',net_loader.nums_class[sess.run(tf.argmax(y_,1),feed_dict={x:fgmask,keep_prob:1.0})[0]])	

                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                                break
                    else:
                            break
                cap.release()
                cv2.destroyAllWindows()
#merging with teja

def test_wtih_cam(net_loader, mirror=False):
    with tf.Session() as sess:
        saver.restore(sess, net_loader.model_dir+ckpt)
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
        #---------------------------------------------------------------------------------------#
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
                target = frame[:256,:256,:]
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
                if mirror == True:
                    op = cv2.flip(op,1)
                cv2.imshow('Actual',op)
                t_=thresh
                if mirror == True:
                    thresh = cv2.flip(thresh,1)
                    t_ = cv2.flip(t_,1)
                thresh = cv2.resize(thresh ,(128,128), interpolation = cv2.INTER_CUBIC)
                thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
                thresh = np.reshape(thresh , [1,128,128, 1])
                thresh_ = np.reshape(thresh , [1,128*128*1])
                ans = net_loader.nums_class[sess.run(tf.argmax(y_,1),feed_dict={x:thresh_,keep_prob:1.0})[0]]

                print('Predicted :',ans)    
                
                op_ = np.zeros((100,256,3))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(op_,ans.split('/')[-1],(0,90), font, 2,(255,0,0),2,cv2.LINE_AA)
                op = np.vstack((t_,op_))
                cv2.imshow('Output', op)
                
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break      

        cap.release()
        cv2.destroyAllWindows()
