import itertools
import random
import sys
import time
from copy import deepcopy
import os

import numpy as np
import tensorflow as tf
from scipy.stats import skew, kurtosis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC
from sktime.classification.deep_learning import LSTMFCNClassifier, CNNClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.deep_learning.macnn import MACNNClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate

import config
import utils
from data import Data
from preprocess import Normalizer
from reporter import BaselineReporter
from utils import Metric
import contreg


def baseline_svm(binary, X_train, y_train, X_test, y_test, train_df):
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


def baseline_minirocket(binary, X_train, y_train, X_test, y_test, train_df):
    minirocket = MiniRocketMultivariate(n_jobs=64)
    minirocket.fit(X_train)
    X_train_transform = minirocket.transform(X_train)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)
    X_test_transform = minirocket.transform(X_test)
    y_test_pred = classifier.predict(X_test_transform)
    return Metric(y_true=y_test, y_pred=y_test_pred, binary=binary)


def baseline_lstmfcn(binary, X_train, y_train, X_test, y_test, train_df):
    n_epochs = 20
    if y_train.shape[0] > 20 * 4096:
        batch_size = 4096
        n_epochs = 100
    elif y_train.shape[0] > 20 * 512:
        batch_size = 512
    else:
        batch_size = 128
    clf = LSTMFCNClassifier(n_epochs=n_epochs, batch_size=batch_size, verbose=True)
    clf.fit(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)


def baseline_cif(binary, X_train, y_train, X_test, y_test, train_df):
    clf = CanonicalIntervalForest(n_estimators=16, n_jobs=64, min_interval=8,
                                  max_interval=12, n_intervals=4,
                                  att_subsample_size=16)
    clf.fit(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)


def baseline_cnn(binary, X_train, y_train, X_test, y_test, train_df):
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


def baseline_macnn(binary, X_train, y_train, X_test, y_test, train_df):
    clf = MACNNClassifier(n_epochs=3)
    clf.fit(X_train, y_train)

    try:
        # Attempt the prediction
        y_test_prediction = clf.predict(X_test)
        print("Prediction successful for this configuration.") # Indicate success

    except Exception as e:
        # Handle any error that occurs during the predict call
        print(f"--- ERROR: Prediction failed for this configuration! ---")
        print(f"Error details: {e}")
        # Create a default prediction array of zeros
        # This creates an array of zeros with the same shape and data type as y_test
        y_test_prediction = np.zeros_like(y_test)
        print(f"Generated default prediction array of shape {y_test_prediction.shape} with all zeros.")
        print("Returning default prediction metric.")

    # Return the Metric object using whatever y_test_prediction ended up being
    # (either the actual prediction or the default zeros array)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)



def baseline_contreg(binary, X_train, y_train, X_test, y_test, train_df):
    y_type_train = train_df["flare_type_num"].to_numpy()
    X_train = X_train.transpose([0, 2, 1])
    X_test = X_test.transpose([0, 2, 1])
    y_test_prediction = contreg.contrastive_regression(X_train, y_train, y_type_train, X_test, y_test, None)
    return Metric(y_true=y_test, y_pred=y_test_prediction, binary=binary)




def cross_val(args, data, reporter, method):
    all_tests_metric = Metric(binary=args.binary)

    start_time = time.time()
    args.test_part = 5
    X_train, y_train, _, _, X_test, y_test, train_df, _, _ = data.numpy_datasets(args, args.run_no)
    test_metric = method(args.binary, X_train, y_train, X_test, y_test, train_df)
    run_time = time.time() - start_time
    print(f"test part {5}: {test_metric}, "
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
    rng = random.Random(args.seed)
    for _ in range(args.n_search):
        args.train_n = rng.choice(config.split_sizes)
        args.train_k = None
        args.nan_mode = rng.choice(config.nan_modes)
        args.normalization_mode = rng.choice(config.normalizations)
        print()
        print(f"Next random search -------------------------------------------")
        print(f"training mode: {'n'}, "
              f"split: {args.train_n}, "
              f"nan_mode: {args.nan_mode}, "
              f"normalization: {args.normalization_mode}")
        multi_run(args, data, method)


def main():
    start = time.time()
    args = utils.arg_parse()
    data = Data(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    elif args.experiment == "contreg":
        method = baseline_contreg
    elif args.experiment == "macnn":
        method = baseline_macnn
    print(method.__name__)
    randomized_search(args, data, method)
    print(f"{(time.time() - start) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
