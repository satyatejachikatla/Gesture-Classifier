import netloader, cv2, cnn, tensorflow as tf
"""
With video check
"""
def foo(net_loader):
        with tf.Session() as sess:
                saver = cnn.saver
                ckpt = 'model1.ckpt'
                saver.restore(sess, net_loader.model_dir+ckpt)
                cap = cv2.VideoCapture(0)
                fgbg = cv2.createBackgroundSubtractorMOG2()
                ret, frame = cap.read()
                while(1):
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
