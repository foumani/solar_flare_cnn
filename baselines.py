import itertools
import random
import time
from copy import deepcopy

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC
from sktime.classification.deep_learning import LSTMFCNClassifier, CNNClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.transformations.panel.rocket import MiniRocketMultivariate

import config
import utils
from data import Data
from preprocess import Normalizer
from reporter import BaselineReporter
from utils import Metric


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
    minirocket = MiniRocketMultivariate(n_jobs=64)
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
    clf = CanonicalIntervalForest(n_estimators=16, n_jobs=64, min_interval=8,
                                  max_interval=12, n_intervals=4,
                                  att_subsample_size=16)
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


def cross_val(args, data, reporter, method):
    all_tests_metric = Metric(binary=args.binary)
    for test_part in range(1, 6):
        start_time = time.time()
        args.test_part = test_part
        X_train, y_train, _, _, X_test, y_test = data.numpy_datasets(args)
        test_metric = method(args.binary, X_train, y_train, X_test, y_test)
        run_time = time.time() - start_time
        print(f"test part {test_part}: {test_metric}, "
              f"run time: {run_time * 1000:.0f} ms")
        all_tests_metric += test_metric
        reporter.split_row(args, test_metric)
        reporter.save_split_report(args, incremental=True)
    print(f"method {method.__name__} all tests: {all_tests_metric}")
    return all_tests_metric


def multi_run(args, data, method):
    args.split_report_filename = f"{method.__name__}-split.csv"
    args.model_report_filename = f"{method.__name__}-model.csv"

    reporter = BaselineReporter()
    utils.reset_seeds(args)
    for i in range(args.runs):
        args.run_no = i
        test_metric = cross_val(args, data, reporter, method)
        reporter.model_row(args, test_metric)
        reporter.save_model_report(args)


def randomized_search(args, data, method):
    print()
    print(f"Next random search -------------------------------------------")
    print(f"training mode: {'n'}, "
          f"split: {args.train_n}, "
          f"nan_mode: {args.nan_mode}, "
          f"normalization: {args.normalization_mode}")

    rng = random.Random(args.seed)
    for _ in range(args.n_search):
        args.train_n = rng.choice(config.split_sizes)
        args.train_k = None
        args.nan_mode = rng.choice(config.nan_modes)
        args.normalization_mode = rng.choice(config.normalizations)
        multi_run(args, data, method)


def main():
    start = time.time()
    args = utils.arg_parse()
    data = Data(args)
    method = None
    if args.experiment == "svm":
        method = baseline_svm
    elif args.experiment == "minirocket":
        method = baseline_minirocket
    elif args.experiment == "lstm":
        method = baseline_lstmfcn
    elif args.experiment == "cif":
        method = baseline_cif
    elif args.experiment == "cnn":
        method = baseline_cnn
    print(method.__name__)
    randomized_search(args, data, method)
    print(f"{(time.time() - start) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
