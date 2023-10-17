import os
import time

import numpy
import numpy as np
import torch

import train
from context import Context
from data import Data


def main():
    multi_context = Context(binary=False,
            batch_size=256,
            train_n=[2000, 2000, 400, 120],
            ch_conv1=64, ch_conv2=32, ch_conv3=0,
            l_hidden=32,
            nan_mode="avg", normalization_mode="scale",
            class_importance=[1, 1, 5, 15],
            lr=0.01, data_dropout=0.2, layer_dropout=0.4,
            val_p=0.5, stop=1000, early_stop=40, draw=False)
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
                             class_importance=[0.3, 0.7], stop=1000,
                             early_stop=40,
                             draw=False,
                             ablation=False)
    binary_context_ablation = Context(binary=True,
                             train_n=[4000, 2200],
                             ch_conv1=64, ch_conv2=2, ch_conv3=0,  # since we don't have any layer after convolution
                             l_hidden=0,
                             nan_mode="avg",
                             batch_size=256,
                             normalization_mode="scale",
                             data_dropout=0.1,
                             layer_dropout=0.1,
                             val_p=0.4,
                             class_importance=[0.3, 0.7], stop=1000,
                             early_stop=40,
                             draw=False,
                             ablation=True)
    context = binary_context_ablation
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
    
    run_vals = []
    for _ in range(10):
        val, test = train.cross_val(context, data, None)
        run_vals.append(test)
    
    np.save(f"./experiments_plot/train_{context.binary}_cm_{context.ablation}.npy",
            np.array([run_val.cm for run_val in run_vals]))


if __name__ == "__main__":
    main()
