import cnn2, netloader
'''
Set up the data and start training or testing.
'''
cfg = open('params.cfg','r')
params = cfg.readlines()
cfg.close()
for i in range(len(params)):
    params[i] = params[i].strip()
nl = netloader.NetLoader(params[0],params[1],params[2])
cnn2.train(int(params[3]),int(params[4]),params[5],nl,reload=params[6])
#cnn.foo1(nl)
