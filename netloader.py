import os, traceback,random, cv2, imageio

class NetLoader:
    def __init__(self, model_dir, train_dir, test_dir):
        self.model_dir = model_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_size = 0
        self.test_size = 0
        self.class_nums = {}
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
        for dirname,dirnames,filenames in os.walk(self.test_dir):
            if dirname != self.test_dir:
                self.class_nums[dirname]=i
                i += 1
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
                if filename.endswith('jpg'):
                        train_data.append([os.path.join(dirname,filename),self.labels[self.class_nums[dirname]]])
                        self.train_size += 1
        for dirname, dirnames, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if filename.endswith('jpg'):
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
        train_array = [cv2.resize(imageio.imread(file)[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3) for file,lab in self.train_data[start:end]]
        train_labels = [lab for file,lab in self.train_data[start:end]]
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
            train_array.append(cv2.resize(imageio.imread(self.train_data[j][0])[:,:,:3],(66,76),interpolation=cv2.INTER_AREA).reshape(76*66*3))    
            train_labels.append(self.train_data[j][1])
            if len(self.files_read) == self.train_size-1:
                self.files_read = {}
        return [train_array, train_labels]

    




                          
                
    
