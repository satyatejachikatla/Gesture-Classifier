import cv2,os
if not os.path.exists('./T_Coloured Data'):
    os.mkdir('./T_Coloured Data')
for dirs, subdirs, files in os.walk('./Coloured Data'):
    i = 0
    for file in files:    
        if i%4 == 0 and file.endswith('.jpg'):
            #print(os.path.basename(os.path.dirname(os.path.join(dirs,file))))
            img = cv2.imread(os.path.join(dirs,file),1)
            if not os.path.exists(os.path.join('./T_Coloured Data',os.path.basename(os.path.dirname(os.path.join(dirs,file))))):
                os.mkdir(os.path.join('./T_Coloured Data',os.path.basename(os.path.dirname(os.path.join(dirs,file)))))
            cv2.imwrite(os.path.join(os.path.join('./T_Coloured Data',os.path.basename(os.path.dirname(os.path.join(dirs,file)))),file),img)
            os.remove(os.path.join(dirs,file))
        i += 1
                

