import os
import time

import numpy
import numpy as np
import torch

import train
from context import Context
from data import Data


def p_tuning(context: Context, data):
    run_vals = []
    for i in range(9):
        context.val_p = i * 0.1 + 0.1
        val, test = train.cross_val(context, data, None)
        run_vals.append(test)
    return run_vals


def lr_tuning(context: Context, data):
    run_vals = []
    for i in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        context.lr = i
        val, test = train.cross_val(context, data, None)
        run_vals.append(test)
    return run_vals


def layer_tuning(context: Context, data):
    run_vals = []
    for i in [4, 8, 16, 32, 64, 128, 256]:
        context.ch_conv2 = i
        val, test = train.cross_val(context, data, None)
        run_vals.append(test)
    return run_vals


def instance_tuning(context: Context, data):
    run_vals = []
    for i in [200, 400, 600, 800, 1000, 1200]:
        context.train_n[1] = i
        val, test = train.cross_val(context, data, None)
        run_vals.append(test)
    return run_vals


def main():
    multi_context = Context(binary=False, batch_size=128,
                            train_n=[1600, 1200, 400, 120],
                            ch_conv1=64, ch_conv2=24, ch_conv3=0,
                            l_hidden=16,
                            nan_mode="avg", normalization_mode="scale",
                            class_importance=[0.1, 0.1, 0.2, 0.6],
                            lr=0.01, data_dropout=0.2, layer_dropout=0.5,
                            val_p=0.5, stop=1000, early_stop=40)
    binary_context = Context(binary=True,
                             train_n=[4000, 2200],
                             ch_conv1=64, ch_conv2=32, ch_conv3=0,
                             l_hidden=0,
                             nan_mode="avg",
                             batch_size=256,
                             normalization_mode="scale",
                             data_dropout=0.1,
                             layer_dropout=0.1,
                             val_p=0.4,
                             lr=0.01,
                             class_importance=[0.3, 0.7], stop=1000,
                             early_stop=40,
                             draw=False,
                             ablation=False)
    context = binary_context
    numpy.set_printoptions(precision=2, formatter={'int': '{:5d}'.format,
                                                   'float': '{:7.2f}'.format})
    start = time.time()
    Context.device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu")
    Context.data_dir = "/home/arash/data/solar_flare"
    Context.tensorlog_dir = "tensorlog"
    Context.log_dir = "log"
    if not os.path.exists(Context.log_dir):
        os.makedirs(Context.log_dir)
    Context.model_dir = "models"
    if not os.path.exists(Context.model_dir):
        os.makedirs(Context.model_dir)
    Context.files_df_filename = "all_files.csv"
    Context.files_np_filename = "full_data_X_1_25.npy"
    data = Data()
    
    # method = p_tuning
    # method = lr_tuning
    # method = layer_tuning
    method = instance_tuning
    print(method.__name__)
    run_vals = method(context, data)
    
    np.save(
        f"./experiments_plot/train_{context.binary}_cm_{method.__name__}.npy",
        np.array([run_val.cm for run_val in run_vals]))


if __name__ == "__main__":
    main()
