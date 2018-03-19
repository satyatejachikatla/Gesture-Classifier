import cv2, os, numpy as np, random

for dirname, dirnames, filenames in os.walk('1/'):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname,filename),0)
        for j in range(10):
            M = np.float32([[1,0,random.randint(-100,100)],[0,1,random.randint(-100,100)]])
            s = random.randint(50,200)/100
            img2 = cv2.resize(img,(0,0),fx=s,fy=s,interpolation=cv2.INTER_AREA)
            img2= cv2.warpAffine(img2,M,(img.shape[0],img.shape[1]))
            cv2.imshow('win',img2)
            cv2.waitKey(1)
