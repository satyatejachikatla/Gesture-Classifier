import cv2, os
cap = cv2.VideoCapture(0)
num_classes = 0
font = cv2.FONT_HERSHEY_SIMPLEX
cont = 'y'
phase = 'Test/'
if not os.path.exists('N_Data'):
    os.makedirs('N_Data')
    os.makedirs('N_Data/Train')
    os.makedirs('N_Data/Test')
while cont == 'y':
    num_classes += 1
    clas = input('Enter the name of class '+str(num_classes))
    if not os.path.exists('N_Data/'+phase+clas):
        os.makedirs('N_Data/'+phase+clas)
    i = 0
    print('Press space to start capture')
    while True:
        ret, frame = cap.read()
        frame = frame[frame.shape[0]//4:frame.shape[0]//4+128,frame.shape[1]//4:frame.shape[1]//4+128]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('win',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    print('press space to stop capture')
    while True:
        i += 1
        ret, frame = cap.read()
        frame = frame[frame.shape[0]//4:frame.shape[0]//4+128,frame.shape[1]//4:frame.shape[1]//4+128]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame2 = frame.copy()
        cv2.putText(frame2,clas+'/'+str(i),(0,20), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.imshow('win',frame2)
        if cv2.waitKey(200) & 0xFF == ord(' '):
            break
        cv2.imwrite('N_Data/'+phase+clas+'/'+str(i)+'.jpg',frame)
    cont = input('continue? (y)')
cv2.destroyAllWindows()
