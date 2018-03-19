import cv2, numpy as np, os, random
i = 0
img2 = np.zeros((128,128),np.float32)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
for dirname, dirnames, filenames in os.walk('1/'):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname,filename),0)
        i += 1
        rows,cols = img.shape
        for j in range(100):
            M = np.float32([[1,0,i+1],[0,1,random.randint(1,j+1)]])
            cl1 = clahe.apply(img)
            cl1 = clahe.apply(cl1)
            dst = cv2.warpAffine(cl1,M,(cols,rows))
##            cv2.imshow('win1',img)
##            cv2.imshow('win',img2)
            img2 += dst        
            #cv2.waitKey(1)


img2 //= i*100
cv2.imwrite('test.jpg',img2)
