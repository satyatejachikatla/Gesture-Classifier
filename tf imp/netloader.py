import os, traceback,random, cv2

class NetLoader:
    def __init__(self, model_dir, train_dir, test_dir):
        self.model_dir = model_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_size = 0
        self.test_size = 0
        self.class_nums = {}
        self.nums_class = {}
        self.num_classes = 0
        self.labels = []
        self.files_read = {}
        self.train_data = []
        self.test_data = []
        i = 0
        for dirname,dirnames,filenames in os.walk(self.train_dir):
            if dirname != self.train_dir:
                self.class_nums[dirname]=i
                i += 1
        for j in range(i):
            l = []
            for k in range(i):
                if k==j:
                    l.append(1)
                else:
                    l.append(0)
            self.labels.append(l)
        i=0
        self.num_classes = len(self.class_nums)
        for dirname,dirnames,filenames in os.walk(self.test_dir):
            if dirname != self.test_dir:
                self.class_nums[dirname]=i
                i += 1
        for key in self.class_nums:
            self.nums_class[self.class_nums[key]] = key
        self.create_data()



    """
    Create the train and test data. Inputs are stored as paths to the images, and expected outputs as one-hot vectors.
    """
    def create_data(self):
        files = []
        train_data = []
        test_data = []
        for dirname, dirnames, filenames in os.walk(self.train_dir):
            for filename in filenames:
                if filename.endswith('jpg') or filename.endswith('pgm'):
                        train_data.append([os.path.join(dirname,filename),self.labels[self.class_nums[dirname]]])
                        self.train_size += 1
        for dirname, dirnames, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if filename.endswith('jpg') or filename.endswith('pgm'):
                    test_data.append([os.path.join(dirname,filename),self.labels[self.class_nums[dirname]]])
                    self.test_size += 1
        self.train_data = train_data
        self.test_data = test_data
        print('train data imgs: ',self.train_size)
        print('test data imgs: ',self.test_size)


    """
    Get training data in a sequential manner
    """
    def get_batch(self,start,end):
        train_array = []
        train_labels = []
        for file,lab in self.train_data[start:end]:
            img = cv2.imread(file,0)
            ch = 1
            if len(img.shape) > 2:
                ch = min(3,img.shape[2])
                img = img[:,:,:ch]
            train_array.extend([cv2.resize(img,interpolation=cv2.INTER_AREA).reshape(128*128*ch)])
            train_labels.extend([lab])
        #print(train_array)
        #print(train_labels)
        return [train_array, train_labels]


    """
    Get training data randomly
    """
    def get_batch_random(self,batch_sz):
        train_array = []
        train_labels = []
        for i in range(batch_sz):
            j = random.randint(0,self.train_size-1)
            while j in self.files_read:
                j = random.randint(0,self.train_size-1)
            img = cv2.imread(self.train_data[j][0],0)
            ch = 1
            if len(img.shape) > 2:
                ch = min(3,img.shape[2])
                img = img[:,:,:ch]
            train_array.append(cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA).reshape(128*128*ch))    
            train_labels.append(self.train_data[j][1])
            if len(self.files_read) == self.train_size:
                self.files_read = {}
        return [train_array, train_labels]

    def get_single_img(self,file):
        img = cv2.imread(file,0)
        ch = 1
        if len(img.shape) > 2:
            ch = min(3,img.shape[2])
            img = img[:,:,:ch]
        ip = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA).reshape(128*128*ch)
        return ip
        
        

    




                          
                
    
