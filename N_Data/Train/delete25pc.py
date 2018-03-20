import os
print(os.path.abspath(os.curdir))
for dirname, dirnames, filenames in os.walk('.'):
    i = 0
    for filename in filenames:
        if not filename.endswith('.jpg'):
            continue
        if i % 4 == 0:
            print(os.path.join(dirname,filename))
            os.remove(os.path.join(dirname,filename))
        i += 1
