def optimal_model(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = [5000, 3500]  # [2250, 1600] # [1400, 1000]
        args.ch_conv1 = 32  # 32
        args.ch_conv2 = 64  # 64
        args.ch_conv3 = 128  # 128
        args.l_hidden1 = 64  # 64
        args.l_hidden2 = 32  # 32
        args.nan_mode = 0
        args.batch_size = 1024  # 1024
        args.normalization_mode = "scale"
        args.data_dropout = 0.3
        args.layer_dropout = 0.3  # 0.3
        args.class_importance = [0.4, 0.6]  # [0.4, 0.6]
        args.val_p = 0.8  # 0.3 # 0.6
        args.run_no = 5
        args.cache = True
        args.kernel_size = [7, 7, 5]
        args.pooling_size = [4, 4, 4]
        args.rand_seed = 42
        args.np_seed = 42
        args.torch_seed = 42
    else:
        args.batch_size = 256
        args.train_n = [2000, 2000, 400, 120]
        args.ch_conv1 = 64
        args.ch_conv2 = 32
        args.ch_conv3 = 0
        args.l_hidden = 32
        args.nan_mode = "avg"
        args.normalization_mode = "scale"
        args.class_importance = [1, 1, 5, 15]
        args.lr = 0.01
        args.data_dropout = 0.2
        args.layer_dropout = 0.4
        args.val_p = 0.5
    return args

def optimal_svm(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = None
        args.train_k = [3200, 1200]
        args.nan_mode = "avg"
        args.normalization_mode = "z_score"
    else:
        args.train_n = [800, 300, 200, 160]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "z_score"
    return args

def optimal_minirocket(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = [400, 200]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "scale"
    else:
        args.train_n = None
        args.train_k = [400, 300, 200, 40]
        args.nan_mode = 0
        args.normalization_mode = "z_score"
    return args

def optimal_lstm(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = None
        args.train_k = [1200, 400]
        args.nan_mode = None
        args.normalization_mode = "scale"
    else:
        args.train_n = [1600, 900, 200, 160]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "z_score"
    return args

def optimal_cif(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = [2000, 1400]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "z_score"
    else:
        args.train_n = [400, 300, 200, 160]
        args.train_k = None
        args.nan_mode = None
        args.normalization_mode = "z_score"
    return args

def optimal_cnn(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = None
        args.train_k = [1600, 400]
        args.nan_mode = 0
        args.normalization_mode = "scale"
    else:
        args.train_n = [400, 900, 600, 160]
        args.train_k = None
        args.nan_mode = "avg"
        args.normalization_mode = "scale"
    return args