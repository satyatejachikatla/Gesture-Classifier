import cv2, os, numpy as np, random

for dirname, dirnames, filenames in os.walk('./'):
    for filename in filenames:
        if not filename.endswith('jpg'):
            continue
        img = cv2.imread(os.path.join(dirname,filename),0)
        for j in range(5):
            M = np.float32([[1,0,random.randint(-100,100)],[0,1,random.randint(-100,100)]])
            s = random.randint(50,200)/100
            img2 = cv2.resize(img,(0,0),fx=s,fy=s)
            img2= cv2.warpAffine(img2,M,(img.shape[0],img.shape[1]))
            cv2.imwrite(os.path.join(dirname,filename.split('.jpg')[0]+'_aug.jpg'),img2)
##            cv2.imshow('win',img2)
##            cv2.waitKey(1)

