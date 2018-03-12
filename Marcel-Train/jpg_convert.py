import imageio, os
for dirname, dirnames, filenames in os.walk('F:/Work/repos/Gesture-Classifier/Marcel-Train/'):
    for file in filenames:
        if file.endswith('ppm'):
            img = imageio.imread(os.path.join(dirname,file))
            imageio.imwrite(os.path.join(dirname,file).split('ppm')[0]+'jpg',img)
