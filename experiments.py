import matplotlib.pyplot as plt
import numpy as np

import baselines
import train
import util
from data import Data
from reporter import Reporter
from util import Metric


def run_experiment(args, data, method):
    run_vals = []
    for _ in range(5):
        test = baselines.cross_val(args, data, None, method)
        run_vals.append(test)
    
    print(f"method {method.__name__} runvals = {run_vals}, "
          f"avg: {np.average([val.tss for val in run_vals])}")

    poster = f"_{args.poster}" if args.poster is not None else ""
    np.save(f"./experiments_plot/{method.__name__}_{args.binary}{poster}_cm.npy",
            np.array([run_val.cm for run_val in run_vals]))


def model_experiment(args, data, n=1, save=True, seeds=[3764, 7078]):
    run_vals = []
    for run_no in range(n):
        args.run_no = run_no
        args.rand_seed = seeds[0]
        args.np_seed = seeds[1]
        if seeds[2] is not None:
            args.torch_seed = seeds[2][run_no]
        val, test = train.cross_val(args, data, Reporter())
        run_vals.append(test)
    
    if save:
        np.save(
            f"./experiments_plot/train_{args.binary}_cm_{args.ablation}.npy",
            np.array([run_val.cm for run_val in run_vals]))

def svm_experiments(args, data, opt_args, seeds=[3764, 7078]):

    if opt_args:
        args.train_n = None
        args.train_k = [3200, 1200]
        args.nan_mode = "avg"
        args.normalization_mode = "z_score"
    else:
        args = update_to_model_opt(args)
    args.binary = True
    args.rand_seed = seeds[0]
    args.np_seed = seeds[1]
    run_experiment(args, data, baselines.baseline_svm)
    
    # args.train_n = [800, 300, 200, 160]
    # args.train_k = None
    # args.nan_mode = 0
    # args.normalization_mode = "z_score"
    # args.binary = False
    # run_experiment(args, data, baselines.baseline_svm)


def minirocket_experiments(args, data, opt_args, seeds=[3764, 7078]):
    if opt_args:
        args.train_n = [400, 200]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "scale"
    else:
        args = update_to_model_opt(args)
    args.binary = True
    args.rand_seed = seeds[0]
    args.np_seed = seeds[1]
    run_experiment(args, data, baselines.baseline_minirocket)
    
    # args.train_n = None
    # args.train_k = [400, 300, 200, 40]
    # args.nan_mode = 0
    # args.normalization_mode = "z_score"
    # args.binary = False
    # run_experiment(args, data, baselines.baseline_minirocket)


def lstm_experiments(args, data, opt_args, seeds=[3764, 7078]):
    if opt_args:
        args.train_n = None
        args.train_k = [1200, 400]
        args.nan_mode = None
        args.normalization_mode = "scale"
    else:
        args = update_to_model_opt(args)
    args.binary = True
    args.rand_seed = seeds[0]
    args.np_seed = seeds[1]
    run_experiment(args, data, baselines.baseline_lstmfcn)
    
    # args.train_n = [1600, 900, 200, 160]
    # args.train_k = None
    # args.nan_mode = 0
    # args.normalization_mode = "z_score"
    # args.binary = False
    # run_experiment(args, data, baselines.baseline_lstmfcn)


def cif_experiments(args, data, opt_args, seeds=[3764, 7078]):
    if opt_args:
        args.train_n = [2000, 1400]
        args.train_k = None
        args.nan_mode = 0
        args.normalization_mode = "z_score"
    else:
        args = update_to_model_opt(args)
    args.binary = True
    args.rand_seed = seeds[0]
    args.np_seed = seeds[1]
    run_experiment(args, data, baselines.baseline_cif)
    
    # args.train_n = [400, 300, 200, 160]
    # args.train_k = None
    # args.nan_mode = None
    # args.normalization_mode = "z_score"
    # args.binary = False
    # run_experiment(args, data, baselines.baseline_cif)


def cnn_experiments(args, data, opt_args, seeds=[3764, 7078]):
    if opt_args:
        args.train_n = None
        args.train_k = [1600, 400]
        args.nan_mode = 0
        args.normalization_mode = "scale"
    else:
        args = update_to_model_opt(args)
    args.binary = True
    args.rand_seed = seeds[0]
    args.np_seed = seeds[1]
    run_experiment(args, data, baselines.baseline_cnn)
    
    # args.train_n = [400, 900, 600, 160]
    # args.train_k = None
    # args.nan_mode = "avg"
    # args.normalization_mode = "scale"
    # args.binary = False
    # run_experiment(args, data, baselines.baseline_cnn)


def optimal_args(args, binary):
    if binary:
        args.train_n = [1400, 1000]
        args.ch_conv1 = 32
        args.ch_conv2 = 64
        args.ch_conv3 = 128
        args.l_hidden1 = 64
        args.l_hidden2 = 32
        args.nan_mode = 0
        args.batch_size = 256
        args.normalization_mode = "scale"
        args.data_dropout = 0.3
        args.layer_dropout = 0.1
        args.class_importance = [0.5, 0.5]
        args.val_p = 0.3
        args.run_no = 5
        args.cache = True
        args.rand_seed = 3764
        args.np_seed = 7078
        args.torch_seeds = [1046, 35030, 92020, 16679, 22678]
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
        args.binary = False
    
    args.draw = False
    args.ablation = False
    return args

def update_to_model_opt(args):
    args.train_k = None
    args.train_n = [1400, 1000]
    args.nan_mode = 0
    args.normalization_mode = "scale"
    return args


def model_experiments(args, data):
    args = optimal_args(args, binary=True)
    model_experiment(args, data, n=5, seeds=[args.rand_seed, args.np_seed, args.torch_seeds])
    
    # args.ablation = True
    # Since we only need binary classification size for the last layer
    # args.ch_conv2 = 2
    # Everything else is same as default binary classification
    # model_experiment(args, data, n=5)
    
    # args = optimal_args(args, binary=False)
    # model_experiment(args, data, n=10)


def tuning_experiments(args, data):
    args = optimal_args(args, binary=True)
    tuning_experiment(args, data, p_tuning)
    args = optimal_args(args, binary=True)
    tuning_experiment(args, data, lr_tuning)


def draw_embeddings_tsne(args, data):
    args = optimal_args(args, binary=True)
    args.draw = True
    model_experiment(args, data, n=1, save=False)
    
    args = optimal_args(args, binary=False)
    args.draw = True
    model_experiment(args, data, n=1, save=False)


def p_tuning(args, data):
    run_vals = []
    for i in range(9):
        args.val_p = i * 0.1 + 0.1
        val, test = train.cross_val(args, data, None)
        run_vals.append(test)
    return run_vals


def lr_tuning(args, data):
    run_vals = []
    for i in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        args.lr = i
        val, test = train.cross_val(args, data, None)
        run_vals.append(test)
    return run_vals


def tuning_experiment(args, data, method):
    run_vals = []
    for run in range(10):
        print(f"------- Running Experiment {run} of {method.__name__} -------")
        single_run_vals = method(args, data)
        run_vals.append(single_run_vals)
    np.save(f"./experiments_plot/train_{args.binary}_cm_{method.__name__}.npy",
            np.array([[run_val.cm for run_val in single_run_vals]
                      for single_run_vals in run_vals]))
    return run_vals


def plot_algorithm_comparisons(binary):
    cif = np.load(f"./experiments_plot/baseline_cif_{binary}_cm.npy")
    cnn = np.load(f"./experiments_plot/baseline_cnn_{binary}_cm.npy")
    lstmfcn = np.load(f"./experiments_plot/baseline_lstmfcn_{binary}_cm.npy")
    minirocket = np.load(
        f"./experiments_plot/baseline_minirocket_{binary}_cm.npy")
    svm = np.load(f"./experiments_plot/baseline_svm_{binary}_cm.npy")
    mine = np.load(f"./experiments_plot/train_{binary}_cm_False.npy")
    
    cif = [Metric(binary=False, cm=cm) for cm in cif]
    cnn = [Metric(binary=False, cm=cm) for cm in cnn]
    lstmfcn = [Metric(binary=False, cm=cm) for cm in lstmfcn]
    minirocket = [Metric(binary=False, cm=cm) for cm in minirocket]
    svm = [Metric(binary=False, cm=cm) for cm in svm]
    mine = [Metric(binary=False, cm=cm) for cm in mine]
    boxes = [[np.average(m.tss) for m in cif],
             [np.average(m.tss) for m in cnn],
             [np.average(m.tss) for m in lstmfcn],
             [np.average(m.tss) for m in minirocket],
             [np.average(m.tss) for m in svm],
             [np.average(m.tss) for m in mine]]
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.7])
    bp = ax.boxplot(boxes)
    ax.set_xticklabels(
        ["CIF", "Sktime CNN", "LSTM-FCN", "Minirocket", "SVM", "Our Model"])
    plt.ylabel("average TSS")
    plt.title(f"Model comparison ({'binary' if binary else 'multi'})")
    plt.savefig(f"plots/algorithm_comparisons_{binary}.jpg")
    plt.show()


def plot_ablation_comparison():
    ablation = np.load("./experiments_plot/train_True_cm_True.npy")
    model = np.load("./experiments_plot/train_True_cm_False.npy")
    boxes = [[Metric(binary=True, cm=cm).tss for cm in ablation],
             [Metric(binary=True, cm=cm).tss for cm in model]]
    
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.set_xticklabels(["Model Without Feature Layer", "Full Model"])
    bp = ax.boxplot(boxes)
    plt.title("Ablation study")
    plt.ylabel("TSS")
    plt.savefig(f"plots/ablation_comparisons.jpg")
    plt.show()


def plot_tuning_experiments():
    data = np.load("./experiments_plot/train_True_cm_p_tuning.npy")
    run_vals = []
    for single_run_vals in data:
        single_run_val = []
        for run_val in single_run_vals:
            single_run_val.append(Metric(binary=True, cm=run_val))
        run_vals.append(single_run_val)
    
    y = []
    for single_run_vals in run_vals:
        y.append([m.tss for m in single_run_vals])
    y = np.array(y)
    y = np.average(y, axis=0)
    
    fig = plt.figure(figsize=(8, 6))
    x_ticks = [round(0.1 * i + 0.1, 2) for i in range(9)]
    x = np.array(list(range(1, len(x_ticks) + 1))) * 0.1
    plt.plot(x, y)
    plt.ylabel("TSS")
    plt.xlabel("Validation fraction of data")
    plt.title("Effect of different portions of data as validation set")
    plt.savefig("plots/p_tuning_experiment.jpg")
    plt.show()
    
    data = np.load("./experiments_plot/train_True_cm_lr_tuning.npy")
    run_vals = []
    for single_run_vals in data:
        single_run_val = []
        for run_val in single_run_vals:
            single_run_val.append(Metric(binary=True, cm=run_val))
        run_vals.append(single_run_val)
        
    y = []
    for single_run_vals in run_vals:
        y.append([m.tss for m in single_run_vals])
    y = np.array(y)
    y = np.average(y, axis=0)
    y = y[1:]
    
    x_ticks = ["0.001", "0.005", "0.01", "0.05", "0.1", "0.5"]
    plt.ylabel("TSS")
    plt.xlabel("Learning Rate")
    plt.bar(x_ticks, y, width=0.5)
    plt.ylim(bottom=0.7, top=0.9)
    plt.title("Effect of different learning rates")
    plt.savefig("plots/lr_tuning_experiment.jpg")
    plt.show()


def main():
    baseline_args = util.baseline_arg_parse()
    model_args = util.train_arg_parse()
    data = Data(baseline_args, verbose=False)
    baseline_args.poster = "same_seed"
    
    args = optimal_args(model_args, binary=True)
    # model_experiment(args, data, n=1)
    # model_experiments(model_args, data)
    # plot_ablation_comparison()
    # svm_experiments(baseline_args, data, opt_args=False)
    # lstm_experiments(baseline_args, data, opt_args=False)
    minirocket_experiments(baseline_args, data, opt_args=False)
    cnn_experiments(baseline_args, data, opt_args=False)
    cif_experiments(baseline_args, data, opt_args=False)
    # plot_algorithm_comparisons(False)
    # plot_algorithm_comparisons(True)
    #
    # tuning_experiments(model_args, data)
    # plot_tuning_experiments()
    #
    # draw_embeddings_tsne(model_args, data)


if __name__ == "__main__":
    main()
