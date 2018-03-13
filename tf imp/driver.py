import cnn, netloader,params
'''
Set up the data and start training or testing.
'''
nl = netloader.NetLoader(params.model_dir,params.train_dir,params.test_dir)
cnn.train(params.epochs,params.batch_size,params.learning_rate,nl,params.reload_model)
#cnn.foo1(nl)
