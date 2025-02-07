import argparse
import os
import random
from collections import namedtuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

DataPair = namedtuple("DataPair", ["X", "y"])

def reset_seeds(args):
    random.seed(args.rand_seed)
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)

def arg_parse(manual=None):
    parser = common_arg_parse(manual)
    parser.add_argument("--learning_rate", dest="lr", default=0.01, type=float,
                        help="Adam optimizer learning rate.")
    parser.add_argument("--early_stop", dest="early_stop", type=int,
                        default=100)
    parser.add_argument("--stop", dest="stop", default=1000, type=int)
    parser.add_argument("--search", dest="n_search", type=int,
                        default=100)
    parser.add_argument("--batch", dest="batch_size", default=256, type=int)
    parser.add_argument("--draw", dest="draw",
                        action="store_true",
                        help="Draw the t-SNE of the last layer of model.")
    parser.add_argument("--valp", dest="val_p", default=0.4, required=False,
                        type=float,
                        help="Portion of data dedicated to validation.")
    parser.add_argument("--gpu", nargs="?", const=0, type=int,
                        help="Run on GPU. Optionally specify GPU id (default: 0 if flag is provided without a value).")
    parser.add_argument('--multi', dest='binary',
                        action="store_false",
                        help='Runs multi-class classification. (not supported)')
    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='How many times a model runs.')
    parser.add_argument("--datadir", dest="data_dir", required=True,
                        help="Location of data directory.")
    parser.add_argument("--logdir", dest="log_dir", required=True,
                        help="Location of log directory.")
    parser.add_argument("--files_csv", dest="files_df_filename",
                        default="all_files.csv",
                        help="Name of the csv database of instances.")
    parser.add_argument("--files_mem", dest="files_np_filename",
                        default="full_data_X_1_25.npy",
                        help="Name of the numpy file with all instances.")
    parser.add_argument("--n", dest="train_n", default=None,
                        help="Distribution of values for train set as "
                             "proportion for each class. "
                             "Input each value for classes seperated by ','. "
                             "Example: 400,300,200,100 ."
                             "Mutually exclusive with 'k'.")
    parser.add_argument("--k", dest="train_k", default=None,
                        help="Distribution of values for train set as "
                             "proportion for each class. "
                             "Input each value for classes seperated by ','. "
                             "Example: 400,300,200,100 ."
                             "Mutually exclusive with 'n'.")
    parser.add_argument("--valpart", dest="val_part", default=None, type=int,
                        help="Partition of SWAN-SF set as validation. "
                             "It is mutually exclusive with valp.")
    parser.add_argument("--cache", dest="cache", action="store_true")

    parser.add_argument("--experiment", dest="experiment", required=False,
                        help="Possible choices are 'svm', 'minirocket', 'cif', 'cnn', and 'lstm'.")
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
    if not os.path.exists("./experiments_plot"):
        os.makedirs("./experiments_plot")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    if args.train_n is not None:
        args.train_n = [int(a) for a in args.train_n.split(",")]
    if args.train_k is not None:
        args.train_k = [int(a) for a in args.train_k.split(",")]

    args.split_report_filename = f"split_report_{'binary' if args.binary else 'multi'}.csv"
    args.model_report_filename = f"seeded_best_model_report_{'binary' if args.binary else 'multi'}.csv"
    args.poster = None


def print_config(args):
    print(f"cpu count: {os.cpu_count()}")
    print(f"device: {args.device}")
    print(f"data dir: {args.data_dir}")
    print(f"csv database: {args.files_df_filename}")
    print(f"mem instances: {args.files_np_filename}")
    print(f"val p: {args.val_p}")
    print(f"early stop: {args.early_stop}")
    print(f"caching: {args.cache}")


def hash_dataset(partitions, k, n, nan_mode, binary):
    if n is not None:
        hash_str = f"parts{partitions}_n_{n}"
    elif k is not None:
        hash_str = f"parts{partitions}_k_{k}"
    else:
        hash_str = f"parts{partitions}_full"

    hash_str += f"_{nan_mode}" if nan_mode is not None else "_None"
    hash_str += f"_binary" if binary else "_multi"
    return hash_str


def hash_name(args):
    hash_str = ""
    if args.train_n is not None:
        hash_str += f"train(n){args.train_n}_"
    elif args.train_k is None:
        hash_str += f"train(full)_"
    else:
        hash_str += f"train(k){args.train_k}_"

    hash_str += f"val[{args.val_part if args.val_part is not None else args.val_p}]_"
    hash_str += f"test[{args.test_part}]"

    hash_str += f"_batch{args.batch_size}"

    hash_str += f"_model{[args.ch_conv1, args.ch_conv2, args.ch_conv3]}"
    hash_str += f"{[args.l_hidden1, args.l_hidden2]}"

    hash_str += f"_{args.nan_mode}" if args.nan_mode is not None else "_None"
    hash_str += f"_do{[args.data_dropout, args.layer_dropout]}"
    hash_str += f"_lr[{args.lr}]"
    hash_str += f"_binary" if args.binary else "_multi"

    return hash_str


def hash_model(args):
    hash_str = ""
    if args.train_n is not None:
        hash_str += f"train(n){args.train_n}_"
    elif args.train_k is None:
        hash_str += f"train(full)_"
    else:
        hash_str += f"train(k){args.train_k}_"

    hash_str += f"_batch{args.batch_size}"

    hash_str += f"_model{[args.ch_conv1, args.ch_conv2, args.ch_conv3]}"
    hash_str += f"{[args.l_hidden1, args.l_hidden2]}"

    hash_str += f"_{args.nan_mode}" if args.nan_mode is not None else "_None"
    hash_str += f"_do{[args.data_dropout, args.layer_dropout]}"
    hash_str += f"_lr[{args.lr}]"
    hash_str += f"_binary" if args.binary else "_multi"

    return hash_str


def possible_labels(binary):
    return np.array([0, 1]) if binary else np.array([0, 1, 2, 3])


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
