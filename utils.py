import argparse
import os
import sys
import random
from collections import namedtuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from preprocess import Normalizer

DataPair = namedtuple("DataPair", ["X", "y"])


def reset_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def arg_parse(manual=None):
    parser = common_arg_parse(manual)
    parser.add_argument("--learning_rate", dest="lr", default=0.01, type=float,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--earlystop", dest="early_stop", default=40, type=int)
    parser.add_argument("--stop", dest="stop", default=200, type=int,
                        help="Maximum number of training iterations.")
    parser.add_argument("--search", dest="n_search", type=int, default=1,
                        help="Number of search iterations for hyperparameter tuning")
    parser.add_argument("--batch", dest="batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--draw", dest="draw", action="store_true",
                        help="Draw a t-SNE visualization of the model's last layer.")
    parser.add_argument("--smote", dest="smote", action="store_true", default=False)
    parser.add_argument("--valp", dest="val_p", default=0.5, required=False,
                        type=float,
                        help="Fraction of data to dedicate to validation.")
    parser.add_argument("--importance", dest="class_importance", default=None,
                        required=False,
                        help="Comma-seperated list specifying the importance of each class. (e.g., 0.4,0.6)")
    parser.add_argument("--plots", dest="plots", default="plots")
    parser.add_argument("--gpu", nargs="?", const=0, type=int,
                        help="Run on GPU. Optionally specify GPU ID (default: 0 if the flag is provided without a value).")
    parser.add_argument('--multi', dest='binary', action="store_false",
                        help="Disable binary classification to run multi-class classification (currently not supported).")
    parser.add_argument("--ndbsr", dest="ndbsr", action="store_true", default=False,
                        help="Enable Near Decision Boundary Sample Removal.")
    parser.add_argument("--aug", dest="aug", action="store_true", default=False,
                        help="Enable augmentation for the data.")
    parser.add_argument('--runs', dest='runs', default=5, type=int,
                        help='Number of times to run the model (default: 5).')
    parser.add_argument("--datadir", dest="data_dir", required=True,
                        help="Path to the data directory.")
    parser.add_argument("--logdir", dest="log_dir", default="log",
                        help="Path to the log directory.")
    parser.add_argument("--files_csv", dest="files_df_filename", default="all_files.csv",
                        help="Filename for the CSV file containing instance metadata (default: 'all_files.csv').")
    parser.add_argument("--files_mem", dest="files_np_filename",
                        default="full_data_X_1_25.npy",
                        help="Filename for the NumPy file with all instance data (default: 'full_data_X_1_25.npy').")
    parser.add_argument("--n", dest="train_n", default="6500,1000",
                        help="Comma-separated list specifying the distribution of training samples per class (e.g., 400,300,200,100). ")
    parser.add_argument("--valpart", dest="val_part", default=None, type=int,
                        help="Partition index of the SWAN-SF dataset to use for validation."
                             " Mutually exclusive with '--valp'.")
    parser.add_argument("--cache", dest="cache", action="store_true",
                        help="Enable caching of data for faster processing.")
    parser.add_argument("--experiment", dest="experiment", required=False,
                        help="Type of experiment to run. "
                             "Options: 'svm', 'minirocket', 'cif', 'cnn', 'lstm'.")
    parser.add_argument("--verbose", dest="verbose", default=5, required=False, type=int,
                        help="Verbosity level for output. Options are:\n"
                             "  0: No output\n"
                             "  1: No consul output\n"
                             "  2: Minimal (only training start/end messages)\n"
                             "  3: Per-run output\n"
                             "  4: Data reading information\n"
                             "  5: Detailed per-epoch output (default: 5)")
    parser.add_argument("--depth", dest="depth", default=None, required=False,
                        help="Comma-seperated list of length 3 specifying the depth of conv blocks (e.g., 5,7,9)")
    # ------- model stuff ------------------
    parser.add_argument("--hidden", dest="hidden", default=None, required=False,
                        help="Comma-seperated list of length 2 specifying the hidden layer sizes (e.g., 32,16)")
    parser.add_argument("--kernelsize", dest="kernel_size", default=None, required=False,
                        help="Comma seperated list of length 3 specifying the kernel size of conv blocks (e.g., 7,7,5)")
    parser.add_argument("--poolingsize",
                        dest="pooling_size",
                        default=None,
                        required=False,
                        type=int,
                        help="Size of the pooling layer.")
    parser.add_argument("--features", dest="n_features", default=24, required=False,
                        type=int)
    parser.add_argument("--nan", dest="nan_mode", default=None, required=False,
                        help="How to handle NAN numbers in data, if not given does nothing. options: none, 0, local_avg, avg")
    parser.add_argument("--norm", dest="normalization_mode", default=None, required=False,
                        help="How to normalize the data. options: scale, zscore")
    parser.add_argument("--seed", dest="seed", default=42, required=False, type=int,
                        help="Random seed.")
    parser.add_argument("--poolingstrat", dest="pooling_strat", default="max",
                        required=False, help="pooling strategy. options: max, mean")
    parser.add_argument("--datadrop", dest="data_dropout", default=0.0, type=float,
                        required=False)
    parser.add_argument("--layerdrop", dest="layer_dropout", default=0.0, type=float,
                        required=False)
    # ------------------ saving stuff -------------------
    parser.add_argument("--splitreport", dest="split_report_filename", default=None,
                        required=False)
    parser.add_argument("--modelreport", dest="model_report_filename", default=None,
                        required=False)
    parser.add_argument("--configreport", dest="config_report_filename",
                        default="configs.csv", required=False)
    parser.add_argument("--resultfilename", dest="results_filename", default="result", required=False,
                        help="Filename to save the results in.")
    parser.add_argument("--resultdir", dest="results_dir", default="log/results",
                        help="Directory to save the results in.")
    args = parser.parse_args()

    initialize(args)
    return args


def common_arg_parse(manual=None):
    parser = argparse.ArgumentParser(description="Solar prediction arguments.")
    if manual is not None:
        parser.data_dir = manual["data_dir"]
        parser.log_dir = manual["log_dir"]
    return parser


def initialize(args):
    if args.gpu is None:
        args.device = "cpu"
    else:
        args.device = f"cuda:{args.gpu}"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not os.path.exists(args.plots):
        os.makedirs(args.plots)

    if args.train_n is not None:
        args.train_n = [int(a) for a in args.train_n.split(",")]
    if args.depth is not None:
        args.depth = [int(a) for a in args.depth.split(",")]
        if len(args.depth) != 3:
            sys.exit("Input 3 comma seperated numbers for depth. e.g. 8,16,32")
    if args.hidden is not None:
        args.hidden = [int(a) for a in args.hidden.split(",")]
    if args.kernel_size is not None:
        args.kernel_size = [int(a) for a in args.kernel_size.split(",")]
        if len(args.kernel_size) != 3:
            sys.exit("Input 3 comma seperated numbers for --kernelsize. e.g., 7,7,5")
    if args.class_importance is not None:
        args.class_importance = [float(a) for a in args.class_importance.split(",")]
        # todo: add checks here
    if args.nan_mode is not None:
        if args.nan_mode == "0":
            args.nan_mode = 0
        elif args.nan_mode == "avg":
            pass
        elif args.nan_mode == "none":
            args.nan_mode = None
    if args.normalization_mode is not None:
        if args.normalization_mode == "scale":
            args.normalization_mode = Normalizer.scale
        elif args.normalization_mode == "zscore":
            args.normalization_mode = Normalizer.z_score
    args.ablation = False
    args.saved_seed = args.seed
    args.saliency = False

    if args.split_report_filename is None:
        args.split_report_filename = f"split_report_{'binary' if args.binary else 'multi'}.csv"
    if args.model_report_filename is None:
        args.model_report_filename = f"seeded_best_model_report_{'binary' if args.binary else 'multi'}.csv"
    args.poster = None
    args.ordering = None


def print_config(args):
    print(f"cpu count: {os.cpu_count()}")
    print(f"device: {args.device}")
    print(f"data dir: {args.data_dir}")
    print(f"results dir: {args.results_dir}")
    print(f"results file: {args.results_filename}")
    print(f"csv database: {args.files_df_filename}")
    print(f"mem instances: {args.files_np_filename}")
    print(f"val p: {args.val_p}")
    print(f"early stop: {args.early_stop}")
    print(f"caching: {args.cache}")


def hash_dataset(partitions, n, nan_mode, binary):
    if n is not None:
        hash_str = f"parts{partitions}_n_{n}"
    else:
        hash_str = f"parts{partitions}_full"

    hash_str += f"_{nan_mode}" if nan_mode is not None else "_None"
    hash_str += f"_binary" if binary else "_multi"
    return hash_str


def hash_name(args):
    hash_str = ""
    if args.train_n is not None:
        hash_str += f"train(n){args.train_n}_"
    else:
        hash_str += f"train(full)_"

    hash_str += f"val[{args.val_part if args.val_part is not None else args.val_p}]_"
    hash_str += f"test[{args.test_part}]"

    hash_str += f"_batch{args.batch_size}"

    hash_str += f"_model{[args.depth[0], args.depth[1], args.depth[2]]}"
    hash_str += f"{[args.hidden[0], args.hidden[1]]}"

    hash_str += f"_{args.nan_mode}" if args.nan_mode is not None else "_None"
    hash_str += f"_do{[args.data_dropout, args.layer_dropout]}"
    hash_str += f"_lr[{args.lr}]"
    hash_str += f"_binary" if args.binary else "_multi"

    return hash_str


def hash_model(args):
    hash_str = ""
    if args.train_n is not None:
        hash_str += f"train(n){args.train_n}_"
    else:
        hash_str += f"train(full)_"

    hash_str += f"_batch{args.batch_size}"

    hash_str += f"_model{[args.depth[0], args.depth[1], args.depth[2]]}"
    hash_str += f"{[args.hidden[0], args.hidden[1]]}"

    hash_str += f"_{args.nan_mode}" if args.nan_mode is not None else "_None"
    hash_str += f"_do{[args.data_dropout, args.layer_dropout]}"
    hash_str += f"_lr[{args.lr}]"
    hash_str += f"_binary" if args.binary else "_multi"

    return hash_str


def possible_labels(binary):
    return np.array([0, 1]) if binary else np.array([0, 1, 2, 3])


def add_results(args, metrics):
    NUM_CLASSES = 2  # we do only binary here
    NUM_RUNS_PER_EXPERIMENT = args.runs
    new_results = np.empty((0, NUM_CLASSES, NUM_CLASSES))
    for metric in metrics:
        new_results = np.append(new_results, np.array([metric.cm]), axis=0)
    print(new_results.shape)
    print(new_results)
    if os.path.exists(os.path.join(args.results_dir, f"{args.results_filename}.npy")):
        results = np.load(f"./{args.results_dir}/{args.results_filename}.npy")
    else:
        results = np.empty((0, NUM_RUNS_PER_EXPERIMENT, NUM_CLASSES, NUM_CLASSES))
    results = np.append(results, np.array([new_results]), axis=0)
    np.save(f"./{args.results_dir}/{args.results_filename}.npy", results)


class Metric:

    def __init__(self, y_true=None, y_pred=None, binary=True, cm=None):
        self.binary = binary
        labels = possible_labels(self.binary)
        if cm is not None:
            self.cm = cm
        elif y_true is None or y_pred is None:
            self.cm = confusion_matrix([], [], labels=labels)
        else:
            self.cm = confusion_matrix(y_true, y_pred, labels=labels)

    @property
    def tn(self):
        if self.binary:
            return self.cm[0][0]
        else:
            return self.cm.sum() - (self.fp + self.fn + self.tp)

    @property
    def fp(self):
        if self.binary:
            return self.cm[0][1]
        else:
            return self.cm.sum(axis=0) - np.diag(self.cm)

    @property
    def fn(self):
        if self.binary:
            return self.cm[1][0]
        else:
            return self.cm.sum(axis=1) - np.diag(self.cm)

    @property
    def tp(self):
        if self.binary:
            return self.cm[1][1]
        else:
            return np.diag(self.cm)

    @property
    def p(self):
        return self.tp + self.fn

    @property
    def n(self):
        return self.tn + self.fp

    @property
    def a(self):
        return self.p + self.n

    @property
    def p_pred(self):
        return self.tp + self.fp

    @property
    def n_pred(self):
        return self.tn + self.fn

    @property
    def accuracy(self):
        try:
            return (self.tp + self.tn) / self.a
        except:
            return 0.0
        # return sklearn.metrics.accuracy_score(self.cm)

    @property
    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except:
            return 0.0

    @property
    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except:
            return 0.0

    @property
    def f1(self):
        try:
            return ((2 * self.precision * self.recall)
                    / (self.precision + self.recall))
        except:
            return 0.0

    @property
    def tpr(self):
        if np.all(self.tp + self.fn > 0):
            return self.tp / (self.tp + self.fn)
        else:
            return 0.0

    @property
    def tnr(self):
        if np.all(self.tn + self.fp > 0):
            return self.tn / (self.tn + self.fp)
        else:
            return 0.0

    @property
    def tss(self):
        if np.all(self.tpr > 0) or np.all(self.tnr > 0):
            return self.tpr + self.tnr - 1
        else:
            return 0.0

    @property
    def hss1(self):
        try:
            return (self.tp + self.tn - self.n) / self.p
        except:
            return 0.0

    @property
    def hss2(self):
        try:
            return ((2 * (self.tp * self.tn - self.fn * self.fp))
                    / (self.p * self.n_pred + self.n * self.p_pred))
        except:
            return 0.0

    @property
    def gs(self):
        try:
            return ((self.tp * (self.p + self.n) - self.p * self.p_pred)
                    / (self.fn * (self.p + self.n) - self.n * self.p_pred))
        except:
            return 0.0

    def __add__(self, other):
        """

        :type other: Metric
        """
        added_metric = Metric(binary=self.binary, cm=self.cm + other.cm)
        return added_metric

    def __repr__(self):
        if self.binary:
            return (f"Metric("
                    f"tss: {self.tss * 100:5.2f}, "
                    f"f1:  {self.f1 * 100:5.2f}, "
                    f"cm: {self.cm.tolist()})")
        else:
            return (f"Metric("
                    f"avg tss: {np.average(self.tss * 100): 5.2f}, "
                    f"tss: {self.tss * 100},"
                    f"cm: {self.cm.tolist()})")

    def __le__(self, other):
        if self.binary:
            return self.tss <= other.tss
        tss_slf = np.nan_to_num(self.tss, True, nan=0.0)
        tss_otr = np.nan_to_num(other.tss, True, nan=0.0)
        return np.average(tss_slf) <= np.average(tss_otr)

    def __lt__(self, other):
        if self.binary:
            return self.tss < other.tss
        tss_slf = np.nan_to_num(self.tss, True, nan=0.0)
        tss_otr = np.nan_to_num(other.tss, True, nan=0.0)
        return np.average(tss_slf) < np.average(tss_otr)
