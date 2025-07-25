from preprocess import Normalizer

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

hiddens = [[8, 8],
           [16, 8], [16, 16],
           [32, 8], [32, 16], [32, 32],
           [64, 8], [64, 16], [64, 32], [64, 64],
           [128, 8], [128, 16], [128, 32], [128, 64], [128, 128]]

depths = [[2, 4, 8], [3, 6, 12], [4, 8, 16],
          [6, 12, 24], [8, 16, 32], [12, 24, 48],
          [16, 32, 64], [24, 48, 96], [32, 64, 128]]


split_sizes = [(i*500, j*500) for i in range(1, 15) for j in range(1, i+1)]

nan_modes = [None, 0, "avg", "local_avg"]

normalizations = [Normalizer.scale, Normalizer.z_score]

importance = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]

def optimal_model(args, binary=None):
    args.binary = True
    args.train_n = [6500, 1000]
    args.kernel_size = [9,9,9]
    args.depth = [2,4,8]
    args.pooling_size = 2
    args.pooling_strat = "max"
    args.hidden = [128, 32]
    args.nan_mode = "local_avg"
    args.normalization_mode = Normalizer.scale
    args.batch_size = 64
    args.data_dropout = 0.0
    args.layer_dropout = 0.3
    args.class_importance = [0.5, 0.5]
    args.val_p = 0.5
    args.seed = 43
    args.cache = True
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.early_stop = 40
    args.stop = 200
    return args


def optimal_svm(args, binary=None): # perfect
    args.binary = True
    args.train_n = [5000, 4500]
    args.nan_mode = "avg"
    args.normalization_mode = Normalizer.z_score
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.seed = 43
    return args


def optimal_minirocket(args, binary=None):
    args.binary = True
    args.train_n = [6000, 500]
    args.nan_mode = "local_avg"
    args.normalization_mode = Normalizer.scale
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.seed = 43
    return args


def optimal_lstm(args, binary=None):
    args.binary = True
    args.train_n = [2500, 2500]
    args.nan_mode = "avg"
    args.normalization_mode = Normalizer.z_score
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.seed = 42
    return args


def optimal_cif(args, binary=None):
    args.binary = True
    args.train_n = [3500, 2000]
    args.nan_mode = None
    args.normalization_mode = Normalizer.scale
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.seed = 43
    return args


def optimal_cnn(args, binary=None):
    args.binary = True
    args.train_n = [3000, 3000]
    args.nan_mode = "local_avg"
    args.normalization_mode = Normalizer.z_score
    args.ndbsr = True
    args.aug = True
    args.smote = True
    args.seed = 43
    return args

def optimal_contreg(args, binary=None):
    args.binary = True
    args.train_n = [3500, 3500]
    args.nan_mode = "local_avg"
    args.normalization_mode = Normalizer.scale
    args.ndbsr = True
    args.aug = True
    args.smote = True
    return args

def optimal_macnn(args, binary=None):
    args.binary = True
    args.train_n = [4000, 3000]
    args.nan_mode = 0
    args.normalization_mode = Normalizer.scale
    args.ndbsr = True
    args.smote = False
    args.augment = True
    return args

def no_preprocess(args):
    args.train_n = None
    args.nan_mode = None
    args.normalization_mode = "skip"
    args.ndbsr = False
    args.smote = False
    args.aug = False
    return args