import numpy as np

import baselines
import train
import util
from data import Data


def run_experiment(args, data, method):
    run_vals = []
    for _ in range(10):
        test = baselines.cross_val(args, data, None, method)
        run_vals.append(test)
    
    print(f"-------- method {method.__name__} "
          f"avg: {np.average([val.tss for val in run_vals])} --------")
    
    np.save(f"./experiments_plot/{method.__name__}_{args.binary}_cm.npy",
            np.array([run_val.cm for run_val in run_vals]))


def model_experiment(args, data):
    run_vals = []
    for _ in range(10):
        val, test = train.cross_val(args, data, None)
        run_vals.append(test)
    
    np.save(
        f"./experiments_plot/train_{args.binary}_cm_{args.ablation}.npy",
        np.array([run_val.cm for run_val in run_vals]))


def svm_experiments(args, data):
    args.train_n = None
    args.train_k = [3200, 1200]
    args.nan_mode = "avg"
    args.normalization_mode = "z_score"
    args.binary = True
    run_experiment(args, data, baselines.baseline_svm)
    
    args.train_n = [800, 300, 200, 160]
    args.train_k = None
    args.nan_mode = 0
    args.normalization_mode = "z_score"
    args.binary = False
    run_experiment(args, data, baselines.baseline_svm)


def minirocket_experiments(args, data):
    args.train_n = [400, 200]
    args.train_k = None
    args.nan_mode = 0
    args.normalization_mode = "scale"
    args.binary = True
    run_experiment(args, data, baselines.baseline_minirocket)
    
    args.train_n = None
    args.train_k = [400, 300, 200, 40]
    args.nan_mode = 0
    args.normalization_mode = "z_score"
    args.binary = False
    run_experiment(args, data, baselines.baseline_minirocket)


def lstm_experiments(args, data):
    args.train_n = None
    args.train_k = [1200, 400]
    args.nan_mode = None
    args.normalization_mode = "scale"
    args.binary = True
    run_experiment(args, data, baselines.baseline_lstmfcn)
    
    args.train_n = [1600, 900, 200, 160]
    args.train_k = None
    args.nan_mode = 0
    args.normalization_mode = "z_score"
    args.binary = False
    run_experiment(args, data, baselines.baseline_lstmfcn)


def cif_experiments(args, data):
    args.train_n = [2000, 1400]
    args.train_k = None
    args.nan_mode = 0
    args.normalization_mode = "z_score"
    args.binary = True
    run_experiment(args, data, baselines.baseline_cif)
    
    args.train_n = [400, 300, 200, 160]
    args.train_k = None
    args.nan_mode = None
    args.normalization_mode = "z_score"
    args.binary = False
    run_experiment(args, data, baselines.baseline_cif)


def cnn_experiments(args, data):
    args.train_n = None
    args.train_k = [1600, 400]
    args.nan_mode = 0
    args.normalization_mode = "scale"
    args.binary = True
    run_experiment(args, data, baselines.baseline_cnn)
    
    args.train_n = [400, 900, 600, 160]
    args.train_k = None
    args.nan_mode = "avg"
    args.normalization_mode = "scale"
    args.binary = False
    run_experiment(args, data, baselines.baseline_cnn)


def model_experiments(args, data):
    args.train_n = [4000, 2200]
    args.ch_conv1 = 64
    args.ch_conv2 = 32
    args.ch_conv3 = 0
    args.l_hidden = 0
    args.nan_mode = "avg"
    args.batch_size = 256
    args.normalization_mode = "scale"
    args.data_dropout = 0.1
    args.layer_dropout = 0.1
    args.val_p = 0.4
    args.class_importance = [0.3, 0.7]
    args.binary = True
    args.draw = False
    model_experiment(args, data)
    
    args.batch_size = 256
    args.train_n = [2000, 2000, 400, 120],
    args.ch_conv1 = 64
    args.ch_conv2 = 32
    args.ch_conv3 = 0
    args.l_hidden = 32,
    args.nan_mode = "avg"
    args.normalization_mode = "scale"
    args.class_importance = [1, 1, 5, 15]
    args.lr = 0.01
    args.data_dropout = 0.2
    args.layer_dropout = 0.4
    args.val_p = 0.5
    args.draw = False
    args.binary = False
    model_experiment(args, data)


def main():
    args = util.baseline_arg_parse()
    data = Data(args, verbose=False)
    # method = None
    # if args.method == "svm":
    #     if args.binary:
    #         args.train_k = [3200, 1200]
    #         args.nan_mode = "avg"
    #         args.normalization_mode = "z_score"
    #     else:
    #         args.train_n = [800, 300, 200, 160]
    #         args.nan_mode = 0
    #         args.normalization_mode = "z_score"
    #     method = baselines.baseline_svm
    # elif args.method == "minirocket":
    #     if args.binary:
    #         args.train_n = [400, 200]
    #         args.nan_mode = 0
    #         args.normalization_mode = "scale"
    #     else:
    #         args.train_k = [400, 300, 200, 40]
    #         args.nan_mode = 0
    #         args.normalization_mode = "z_score"
    #     method = baselines.baseline_minirocket
    # elif args.method == "lstm":
    #     if args.binary:
    #         args.train_k = [1200, 400]
    #         args.nan_mode = None
    #         args.normalization_mode = "scale"
    #     else:
    #         args.train_n = [1600, 900, 200, 160]
    #         args.nan_mode = 0
    #         args.normalization_mode = "z_score"
    #     method = baselines.baseline_lstmfcn
    # elif args.method == "cif":
    #     if args.binary:
    #         args.train_n = [2000, 1400]
    #         args.nan_mode = 0
    #         args.normalization_mode = "z_score"
    #     else:
    #         args.train_n = [400, 300, 200, 160]
    #         args.nan_mode = None
    #         args.normalization_mode = "z_score"
    #     method = baselines.baseline_cif
    # elif args.method == "cnn":
    #     if args.binary:
    #         args.train_k = [1600, 400]
    #         args.nan_mode = 0
    #         args.normalization_mode = "scale"
    #     else:
    #         args.train_n = [400, 900, 600, 160]
    #         args.nan_mode = "avg"
    #         args.normalization_mode = "scale"
    #     method = baselines.baseline_cnn
    #
    # run_vals = []
    # for _ in range(10):
    #     test = baselines.cross_val(args, data, None, method)
    #     run_vals.append(test)
    #
    # print(
    #     f"runvals = {run_vals}, avg: {np.average([val.tss for val in run_vals])}")
    # np.save(f"./experiments_plot/{method.__name__}_{args.binary}_cm.npy",
    #         np.array([run_val.cm for run_val in run_vals]))
    # lstm_experiments(args, data)
    # svm_experiments(args, data)
    # minirocket_experiments(args, data)
    cif_experiments(args, data)
    # cnn_experiments(args, data)
    # model_experiments(args, data)


if __name__ == "__main__":
    main()
