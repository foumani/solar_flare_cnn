import baselines
from context import Context
import util
from data import Data
import numpy as np


def main():
    Context.data_dir = "/home/arash/data/solar_flare"
    Context.files_df_filename = "all_files.csv"
    Context.files_np_filename = "full_data_X_1_25.npy"
    Context.log_dir = "/home/arash/workspace/solar_flare/log"
    prog_args = util.arg_parse()
    context = Context(run_times=prog_args.run_times,
                      binary=prog_args.binary)
    data = Data(verbose=False)
    method = None
    if prog_args.method == "svm":
        if context.binary:
            context.train_k = [3200, 1200]
            context.nan_mode = "avg"
            context.normalization_mode = "z_score"
        else:
            context.train_n = [800, 300, 200, 160]
            context.nan_mode = 0
            context.normalization_mode = "z_score"
        method = baselines.baseline_svm
    elif prog_args.method == "minirocket":
        if context.binary:
            context.train_n = [400, 200]
            context.nan_mode = 0
            context.normalization_mode = "scale"
        else:
            context.train_k = [400, 300, 200, 40]
            context.nan_mode = 0
            context.normalization_mode = "z_score"
        method = baselines.baseline_minirocket
    elif prog_args.method == "lstm":
        if context.binary:
            context.train_k = [1200, 400]
            context.nan_mode = None
            context.normalization_mode = "scale"
        else:
            context.train_n = [1600, 900, 200, 160]
            context.nan_mode = 0
            context.normalization_mode = "z_score"
        method = baselines.baseline_lstmfcn
    elif prog_args.method == "cif":
        if context.binary:
            context.train_n = [2000, 1400]
            context.nan_mode = 0
            context.normalization_mode = "z_score"
        else:
            context.train_n = [400, 300, 200, 160]
            context.nan_mode = None
            context.normalization_mode = "z_score"
        method = baselines.baseline_cif
    elif prog_args.method == "cnn":
        if context.binary:
            context.train_k = [1600, 400]
            context.nan_mode = 0
            context.normalization_mode = "scale"
        else:
            context.train_n = [400, 900, 600, 160]
            context.nan_mode = "avg"
            context.normalization_mode = "scale"
        method = baselines.baseline_cnn
    
    run_vals = []
    for _ in range(10):
        test = baselines.cross_val(context, data, None, method)
        run_vals.append(test)
    
    print(f"runvals = {run_vals}, avg: {np.average([val.tss for val in run_vals])}")
    np.save(f"./experiments_plot/{method.__name__}_{context.binary}_cm.npy",
            np.array([run_val.cm for run_val in run_vals]))


if __name__ == "__main__":
    main()
