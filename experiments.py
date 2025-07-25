import matplotlib.pyplot as plt
import numpy as np
import torch

import baselines
import config
import train
import utils
from data import Data, data_columns
from reporter import Reporter, BaselineReporter
from utils import Metric
import time
import warnings
import os


def run_experiment(args, data, reporter, method):
    run_vals = []
    for run_no in range(args.runs):
        args.run_no = run_no
        test = baselines.cross_val(args, data, reporter, method)
        run_vals.append(test)

    print(f"method {method.__name__} runvals = {run_vals}, "
          f"avg: {np.average([val.tss for val in run_vals])}")

    poster = f"_{args.poster}" if args.poster is not None else ""
    np.save(f"./{args.results_dir}/{method.__name__}_{args.binary}{poster}_cm.npy",
            np.array([run_val.cm for run_val in run_vals]))
    tsses = [run.tss for run in run_vals]
    avg_tss = np.average(tsses)
    print(f"\nAverage TSS of {args.runs} runs: {avg_tss:.4f}")


def model_experiment(args, data, n=1, save=True):
    run_vals = []
    for run_no in range(n):
        args.run_no = run_no
        _, val, test = train.cross_val(args, data, Reporter())
        run_vals.append(test)

    if save:
        np.save(f"./{args.results_dir}/train_{args.binary}_cm_{args.ablation}.npy",
                np.array([run_val.cm for run_val in run_vals]))


def svm_experiments(args, data, reporter, opt_args):
    if opt_args:
        args = config.optimal_svm(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_svm)


def minirocket_experiments(args, data, reporter, opt_args):
    if opt_args:
        args = config.optimal_minirocket(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_minirocket)


def lstm_experiments(args, data, reporter, opt_args):
    if opt_args:
        args = config.optimal_lstm(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_lstmfcn)


def cif_experiments(args, data, reporter, opt_args):
    if opt_args:
        args = config.optimal_cif(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_cif)


def cnn_experiments(args, data, reporter, opt_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if opt_args:
        args = config.optimal_cnn(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_cnn)


def macnn_experiments(args, data, reporter, opt_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if opt_args:
        args = config.optimal_macnn(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_macnn)


def contreg_experiments(args, data, reporter, opt_args):
    if opt_args:
        args = config.optimal_contreg(args, True)
    else:
        args = config.no_preprocess(args)
    utils.reset_seeds(args)
    run_experiment(args, data, reporter, baselines.baseline_contreg)


def no_preprocessing_experiments(args, data, baseline_reporter, opt_args=True):
    utils.reset_seeds(args)
    args.poster = "nopreprocess"

    # svm
    # svm_experiments(args, data, baseline_reporter, opt_args=False)

    # lstmfcn
    lstm_experiments(args, data, baseline_reporter, opt_args=False)

    # minirocket
    minirocket_experiments(args, data, baseline_reporter, opt_args=False)

    # cnn
    cnn_experiments(args, data, baseline_reporter, opt_args=False)

    # cif
    cif_experiments(args, data, baseline_reporter, opt_args=False)

    # contreg
    contreg_experiments(args, data, baseline_reporter, opt_args=False)

    # macnn
    args.runs = 10
    macnn_experiments(args, data, baseline_reporter, opt_args=False)


def model_experiments(args, data):
    args = config.optimal_model(args, binary=True)
    utils.reset_seeds(args)
    model_experiment(args, data, n=args.runs)

    # args.ablation = True
    # Since we only need binary classification size for the last layer
    # args.ch_conv2 = 2
    # Everything else is same as default binary classification
    # model_experiment(args, data, n=5)

    # args = config.optimal_model(args, binary=False)
    # model_experiment(args, data, n=10)


def draw_embeddings_tsne(args, data):
    args = config.optimal_model(args, binary=True)
    args.draw = True
    model_experiment(args, data, n=1, save=False)

    args = config.optimal_model(args, binary=False)
    args.draw = True
    model_experiment(args, data, n=1, save=False)


def tuning_experiment(args, data, method, n=10):
    run_vals = []
    for run in range(n):
        print(f"------- Running Experiment {run} of {method.__name__} -------")
        single_run_vals = method(args, data)
        run_vals.append(single_run_vals)
    np.save(f"./{args.results_dir}/train_{args.binary}_cm_{method.__name__}.npy",
            np.array([[run_val.cm for run_val in single_run_vals]
                      for single_run_vals in run_vals]))
    return run_vals


def plot_algorithm_comparisons(args, binary):
    cif = np.load(f"./{args.results_dir}/baseline_cif_{binary}_cm.npy")
    cnn = np.load(f"./{args.results_dir}/baseline_cnn_{binary}_cm.npy")
    lstmfcn = np.load(f"./{args.results_dir}/baseline_lstmfcn_{binary}_cm.npy")
    minirocket = np.load(
        f"./{args.results_dir}/baseline_minirocket_{binary}_cm.npy")
    svm = np.load(f"./{args.results_dir}/baseline_svm_{binary}_cm.npy")
    mine = np.load(f"./{args.results_dir}/train_{binary}_cm_False.npy")

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


def plot_ablation_comparison(args):
    ablation = np.load(f"./{args.results_dir}/train_True_cm_True.npy")
    model = np.load(f"./{args.results_dir}/train_True_cm_False.npy")
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


def plot_tuning_experiments(args):
    data = np.load(f"./{args.results_dir}/train_True_cm_p_tuning.npy")
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

    data = np.load(f"./{args.results_dir}/train_True_cm_lr_tuning.npy")
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


def feature_selection(args, data, reporter):
    saliency = np.load("saliency.npy")
    saliency_sum = np.sum(saliency, axis=1)
    feature_names = data_columns(args)[1:25]
    ordering = sorted(range(len(saliency_sum)), key=lambda i: saliency_sum[i],
                      reverse=True)
    args.ordering = ordering
    for i in range(1, 24 + 1):
        config.optimal_model(args, binary=True)
        args.n_features = i
        train.config_run(args, data, reporter)


def main():
    args = utils.arg_parse()
    data = Data(args, verbose=False)
    reporter = Reporter()
    baseline_reporter = BaselineReporter()
    start = time.time()
    warnings.filterwarnings("ignore", category=UserWarning,
                            module=r"torch\.nn\.modules\.lazy")
    # model_experiment(args, data, n=1)
    # model_experiments(model_args, data)
    # plot_ablation_comparison()
    # plot_algorithm_comparisons(False)
    # plot_algorithm_comparisons(True)
    #
    # tuning_experiments(model_args, data)
    # plot_tuning_experiments()
    #
    # draw_embeddings_tsne(model_args, data)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.experiment == "cmod":
        args = config.optimal_model(args, True)
        train.config_run(args, data, reporter)
    if args.experiment == "model":
        train.config_run(args, data, reporter)
    if args.experiment == "search":  # TODO: fix
        train.search(args, data, reporter)
    if args.experiment == "orion_search":  # TODO: fix
        train.bohb_search(args, data, reporter)
    if args.experiment == "saliency":  # TODO: fix
        train.saliency_map(args, data, reporter)
    if args.experiment == "feature_selection":  # TODO: fix
        feature_selection(args, data, reporter)
    if args.experiment == "ablation":  # TODO: fix
        train.ablation(args, data, reporter)
    if args.experiment == "model_experiments":  # TODO: fix
        model_experiments(args, data)
    if args.experiment == "svm":
        args.method_name = baselines.baseline_svm
        svm_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "lstm":  # TODO: fix
        args.method_name = baselines.baseline_lstmfcn
        lstm_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "minirocket":  # TODO: fix
        args.method_name = baselines.baseline_minirocket
        minirocket_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "cnn":  # TODO: fix
        args.method_name = baselines.baseline_cnn
        cnn_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "cif":  # TODO: fix
        args.method_name = baselines.baseline_cif
        cif_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "contreg":  # TODO: fix
        args.method_name = baselines.baseline_contreg
        contreg_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "macnn":  # TODO: fix
        args.method_name = baselines.baseline_macnn
        macnn_experiments(args, data, baseline_reporter, opt_args=True)
    if args.experiment == "no_preprocessing":  # TODO: fix
        no_preprocessing_experiments(args, data, baseline_reporter)
    if args.experiment == "plot":  # TODO: fix
        data.plot_ar_hist(args)
        # data.plot_instance_removal(args)

    reporter.experiment.time(args, start, time.time())


if __name__ == "__main__":
    main()
