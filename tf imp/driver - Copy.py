import cnn2, netloader,params
'''
Set up the data and start training or testing.
'''
nl = netloader.NetLoader(params.model_dir,params.train_dir,params.test_dir)
cnn2.train(params.epochs,params.batch_size,params.learning_rate,nl,params.reload_model)
#cnn2.foo(nl)
#cnn2.test_wtih_cam(nl)
