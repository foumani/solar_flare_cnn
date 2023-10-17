import argparse
from collections import namedtuple

import numpy as np
from sklearn.metrics import confusion_matrix

from context import Context

DataPair = namedtuple("DataPair", ["X", "y"])


def arg_parse():
    parser = argparse.ArgumentParser(description="Solar prediction arguments.")
    parser.add_argument('--binary', dest='binary',
                        action="store_true",
                        help='Whether train for binary or multi classification.')
    parser.add_argument('--runs', dest='run_times', default=1, type=int,
                        help='How many times a model runs.')
    parser.add_argument("--method", dest="method", required=False,
                        help="Running what baseline. Possible choices are 'svm', 'minirocket'")
    parser.add_argument('--paramsearch', dest="n_param_search", type=int, required=False,
                        help="How many random values searched.")
    return parser.parse_args()


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


def hash_name(context: Context):
    hash_str = ""
    if context.train_n is not None:
        hash_str += f"train(n){context.train_n}_"
    elif context.train_k is None:
        hash_str += f"train(full)_"
    else:
        hash_str += f"train(k){context.train_k}_"
    
    hash_str += f"val[{context.val_part if context.val_part is not None else context.val_p}]_"
    hash_str += f"test[{context.test_part}]"
    
    hash_str += f"_batch{context.batch_size}"
    
    hash_str += f"_model{[context.ch_conv1, context.ch_conv2, context.ch_conv3]}"
    hash_str += f"{[context.l_hidden]}"
    
    hash_str += f"_{context.nan_mode}" if context.nan_mode is not None else "_None"
    hash_str += f"_do{[context.data_dropout, context.layer_dropout]}"
    hash_str += f"_lr[{context.lr}]"
    hash_str += f"_binary" if context.binary else "_multi"
    
    return hash_str


def hash_model(context: Context):
    hash_str = ""
    if context.train_n is not None:
        hash_str += f"train(n){context.train_n}_"
    elif context.train_k is None:
        hash_str += f"train(full)_"
    else:
        hash_str += f"train(k){context.train_k}_"
    
    hash_str += f"_batch{context.batch_size}"
    
    hash_str += f"_model{[context.ch_conv1, context.ch_conv2, context.ch_conv3]}"
    hash_str += f"{[context.l_hidden]}"
    
    hash_str += f"_{context.nan_mode}" if context.nan_mode is not None else "_None"
    hash_str += f"_do{[context.data_dropout, context.layer_dropout]}"
    hash_str += f"_lr[{context.lr}]"
    hash_str += f"_binary" if context.binary else "_multi"
    
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
                    f"hss2: {self.hss2 * 100: 5.2f}, "
                    f"cm: {self.cm.tolist()})")
        else:
            return (f"Metric("
                    f"avg tss: {np.average(self.tss * 100): 5.2f}, "
                    f"tss: {self.tss * 100},"
                    f"hss2: {self.hss2 * 100} "
                    f"cm: {self.cm.tolist()})")
    
    # def __str__(self, full=False):
    #     return (f"accuracy {self.accuracy * 100:5.2f}% | "
    #             f"precision {self.precision * 100:6.2f} | "
    #             f"recall {self.recall * 100:6.2f} | "
    #             f"f1 {self.f1 * 100:6.2f} | "
    #             f"TPR {self.tpr * 100:6.2f} | "
    #             f"TNR {self.tnr * 100:6.2f} | "
    #             f"TSS {self.tss * 100:6.2f} | "
    #             f"HSS1 {self.hss1 * 100:8.2f} | "
    #             f"HSS2 {self.hss2 * 100:8.2f} | "
    #             f"GS {self.gs * 100:8.2f} | "
    #             f"tp {self.tp:5.0f}, "
    #             f"fp {self.fp:5.0f}, "
    #             f"tn {self.tn:5.0f}, "
    #             f"fn {self.fn:5.0f}")
    
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
