filter_sizes = [[5, 5, 5], [5, 5, 7], [5, 5, 9],
                [5, 7, 5], [5, 7, 7], [5, 7, 9],
                [5, 9, 5], [5, 9, 7], [5, 9, 9],

                [7, 5, 5], [7, 5, 7], [7, 5, 9],
                [7, 7, 5], [7, 7, 7], [7, 7, 9],
                [7, 9, 5], [7, 9, 7], [7, 9, 9],

                [9, 5, 5], [9, 5, 7], [9, 5, 9],
                [9, 7, 5], [9, 7, 7], [9, 7, 9],
                [9, 9, 5], [9, 9, 7], [9, 9, 9]]

poolings = [[2, "mean"], [2, "max"],
            [3, "mean"], [3, "max"],
            [4, "mean"], [4, "max"]]

hiddens = [[8, 8], [8, 16], [8, 32], [8, 64],
           [16, 8], [16, 16], [16, 32], [16, 64],
           [32, 8], [32, 16], [32, 32], [32, 64],
           [64, 8], [64, 16], [64, 32], [64, 64]]

depths = [[2, 4, 6], [3, 6, 9], [4, 8, 12],
          [6, 12, 18], [8, 16, 24], [12, 24, 36],
          [16, 32, 48], [20, 40, 60], [30, 60, 90],

          [2, 4, 8],    [3, 6, 12],    [4, 8, 16],
          [6, 12, 24],  [8, 16, 32],  [12, 24, 48],
          [16, 32, 64], [20, 40, 80], [30, 60, 120]]


def optimal_model(args, binary=None):
    if binary is None:
        binary = args.binary
    args.binary = binary
    if binary:
        args.train_n = [5000, 3500]  # [2250, 1600] # [1400, 1000]
        args.depth = [32, 64, 128]  # [32, 64, 128]
        args.hidden = [32, 64]  # [32, 64]
        args.nan_mode = 0
        args.batch_size = 1024  # 1024
        args.normalization_mode = "scale"
        args.data_dropout = 0.3
        args.layer_dropout = 0.3  # 0.3
        args.class_importance = [0.4, 0.6]  # [0.4, 0.6]
        args.val_p = 0.8  # 0.3 # 0.6
        args.runs = 1
        args.cache = True
        args.kernel_size = [7, 7, 5]
        args.pooling_size = 4
        args.pooling_strat = "max"
        args.seed
    else:
        args.batch_size = 256
        args.train_n = [2000, 2000, 400, 120]
        args.depth = [64, 32, 0]
        args.hidden = [32, 0]
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
