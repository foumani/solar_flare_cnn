import itertools
import os
import random
import time
from copy import deepcopy

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sktime.classification.deep_learning import LSTMFCNClassifier, CNNClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.transformations.panel.rocket import MiniRocketMultivariate

import util
from context import Context
from data import Data
from preprocess import Normalizer
from reporter import BaselineReporter
from util import Metric


def baseline_svm(binary, X_train, y_train, X_test, y_test):
    def extract_features(data):
        features = np.empty(shape=(len(data), 24, 5))
        features[:, :, 0] = np.median(data, axis=2)
        features[:, :, 1] = np.std(data, axis=2)
        features[:, :, 2] = skew(data, axis=2)
        features[:, :, 3] = kurtosis(data, axis=2)
        features[:, :, 4] = deepcopy(data[:, :, -1])
        features = features.reshape((len(data), 24 * features.shape[2]))
        return features
    
    def preprocess(X, y):
        X = extract_features(X)
        indices = ~np.isnan(X).any(axis=1)
        return X[indices], y[indices]
    
    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)
    svc = SVC(max_iter=2500)
    svc.fit(X_train, y_train)
    y_test_pred = svc.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_pred, binary=binary)


def baseline_minirocket(binary, X_train, y_train, X_test, y_test):
    minirocket = MiniRocketMultivariate(n_jobs=24)
    minirocket.fit(X_train)
    X_train_transform = minirocket.transform(X_train)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)
    X_test_transform = minirocket.transform(X_test)
    y_test_pred = classifier.predict(X_test_transform)
    return Metric(y_true=y_test, y_pred=y_test_pred, binary=binary)


def baseline_lstmfcn(binary, X_train, y_train, X_test, y_test):
    if y_train.shape[0] > 20 * 4096:
        batch_size = 4096
    elif y_train.shape[0] > 20 * 512:
        batch_size = 512
    else:
        batch_size = 128
    clf = LSTMFCNClassifier(n_epochs=20, batch_size=batch_size, verbose=False)
    clf.fit(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)


def baseline_cif(binary, X_train, y_train, X_test, y_test):
    clf = CanonicalIntervalForest(n_estimators=16, n_jobs=16, min_interval=8,
                                  max_interval=12, n_intervals=4, att_subsample_size=16)
    clf.fit(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)


def baseline_cnn(binary, X_train, y_train, X_test, y_test):
    if y_train.shape[0] > 20 * 4096:
        batch_size = 4096
    elif y_train.shape[0] > 20 * 512:
        batch_size = 512
    else:
        batch_size = 128
    clf = CNNClassifier(n_epochs=40, batch_size=batch_size, verbose=False)
    clf.fit(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)


def cross_val(context, data, reporter, method):
    all_tests_metric = Metric(binary=context.binary)
    for test_part in range(1, 6):
        start_time = time.time()
        context.test_part = test_part
        X_train, y_train, _, _, X_test, y_test = data.numpy_datasets(context)
        test_metric = method(context.binary, X_train, y_train, X_test, y_test)
        run_time = time.time() - start_time
        print(
            f"test part {test_part}: {test_metric}, run time: {run_time * 1000:.0f} ms")
        all_tests_metric += test_metric
        if reporter is not None:
            reporter.split_row(context, test_metric)
    print(f"method {method.__name__} all tests: {all_tests_metric}")
    return all_tests_metric


def multi_run(context, data, method, report=True):
    binary_str = "binary" if context.binary else "multi"
    split_report_filename = f"{method.__name__}-split-{binary_str}.csv"
    model_report_filename = f"{method.__name__}-model-{binary_str}.csv"
    if report:
        reporter = BaselineReporter(split_report_filename,
                                    model_report_filename)
    else:
        reporter = None
    for i in range(context.run_times):
        context.run_no = i
        test_metric = cross_val(context, data, reporter, method)
        reporter.model_row(context, test_metric)
    if report:
        reporter.save_model_report(incremental=True)
        reporter.save_split_report(incremental=True)


def randomized_search(context, data, method):
    dataset_grid = []
    training_modes = ["k", "n"]  # 2
    if context.binary:
        bcq = [i for i in range(400, 4001, 400)]  # 10
        mx = [i for i in range(200, 1401, 200)]  # 7
        dataset_grid.extend(list(itertools.product(bcq, mx)))
        class_importances = [None,
                             [0.4, 0.6],
                             [0.3, 0.7],
                             [0.2, 0.8]]  # 4
    else:
        q = [i for i in range(400, 1601, 400)]  # 4
        bc = [i for i in range(300, 1201, 300)]  # 4
        m = [i for i in range(200, 801, 200)]  # 4
        x = [i for i in range(40, 181, 40)]  # 4
        dataset_grid.extend(list(itertools.product(q, bc, m, x)))
    nan_modes = [0, None, "avg"]
    normalizations = [Normalizer.scale, Normalizer.z_score]
    grid = list(itertools.product(training_modes, dataset_grid, nan_modes,
                                  normalizations))
    randomized = random.sample(grid, k=context.n_random_search)
    print(f"Searching through {len(randomized)} items")
    for training_mode, split, nan_mode, normalization in randomized:
        print()
        print(f"Next random search -------------------------------------------")
        print(f"training mode: {training_mode}, "
              f"split: {split}, "
              f"nan_mode: {nan_mode}, "
              f"normalization: {normalization}")
        if training_mode == "n":
            context.train_n = split
            context.train_k = None
        else:
            context.train_k = split
            context.train_n = None
        context.nan_mode = nan_mode
        context.normalization_mode = normalization
        multi_run(context, data, method)


def main():
    start = time.time()
    print(f"cpu count: {os.cpu_count()}")
    Context.data_dir = "/home/foumani/data/solar_flare_prediction"
    Context.files_df_filename = "all_files.csv"
    Context.files_np_filename = "full_data_X_1_25.npy"
    Context.log_dir = "/home/foumani/workspace/solar_flare_prediction/log"
    prog_args = util.arg_parse()
    context = Context(run_times=prog_args.run_times,
                      binary=prog_args.binary,
                      n_random_search=prog_args.n_param_search)
    data = Data(verbose=False)
    method = None
    if prog_args.method == "svm":
        method = baseline_svm
    elif prog_args.method == "minirocket":
        method = baseline_minirocket
    elif prog_args.method == "lstm":
        method = baseline_lstmfcn
    elif prog_args.method == "cif":
        method = baseline_cif
    elif prog_args.method == "cnn":
        method = baseline_cnn
    print(method.__name__)
    randomized_search(context, data, method)
    print(f"{(time.time() - start) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
