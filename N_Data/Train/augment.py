import cv2, os, numpy as np, random

for dirname, dirnames, filenames in os.walk('./'):
    for filename in filenames:
        if not filename.endswith('jpg'):
            continue
        img = cv2.imread(os.path.join(dirname,filename),0)
        img = ~img
        for j in range(10):
            M = np.float32([[1,0,random.randint(-25,25)],[0,1,random.randint(-25,25)]])
            s = random.randint(90,110)/100
            img2 = cv2.resize(img,(0,0),fx=s,fy=s,interpolation=cv2.INTER_AREA)
            img2= cv2.warpAffine(img2,M,(img.shape[0],img.shape[1]))
            img2 = ~img2
            cv2.imwrite(os.path.join(dirname,filename.split('.jpg')[0]+'_a.jpg'),img2)
            
