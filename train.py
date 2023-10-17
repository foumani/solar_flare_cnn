import itertools
import os
import random

import torch.nn as nn
from matplotlib.colors import ListedColormap

import util
from algorithm import *
from context import Context
from conv_model import ConvModel
from data import Data
from preprocess import Normalizer
from reporter import Reporter


def draw(emb, y, hash_name, binary):
    tsne = TSNE(n_components=2 if binary else 4,
                method="barnes_hut" if binary else "exact")
    plt.figure(figsize=(5, 3))
    
    if binary:
        qbc_idx = np.random.choice((y == 0).nonzero()[0], 1500, replace=False)
        mx_idx = np.random.choice((y == 1).nonzero()[0], 500, replace=False)
        tsne_results = tsne.fit_transform(np.append(emb[qbc_idx],
                                                    emb[mx_idx],
                                                    axis=0))
        y_emb = np.append(np.zeros(qbc_idx.shape[0]), np.ones(mx_idx.shape[0]))
    else:
        q_idx = np.random.choice((y == 0).nonzero()[0], 400, replace=False)
        bc_idx = np.random.choice((y == 1).nonzero()[0], 300, replace=False)
        m_idx = np.random.choice((y == 2).nonzero()[0], 200, replace=False)
        x_idx = np.random.choice((y == 3).nonzero()[0], 100, replace=True)
        tsne_results = tsne.fit_transform(np.concatenate((emb[q_idx],
                                                          emb[bc_idx],
                                                          emb[m_idx],
                                                          emb[x_idx]),
                                                         axis=0))
        y_emb = np.concatenate((np.zeros(q_idx.shape[0]),
                                np.ones(bc_idx.shape[0]),
                                np.full(m_idx.shape[0], 2),
                                np.full(x_idx.shape[0], 3)), axis=0)
    if not binary:
        colors = ListedColormap(['#f1948a', '#e74c3c', '#b03a2e', '#78281f'])
    else:
        colors = ListedColormap(['#f1948a', '#78281f'])
    sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_emb,
                     cmap=colors, s=4)
    if binary:
        plt.legend(handles=sc.legend_elements()[0], labels=["QBC", "MX"])
    else:
        plt.legend(handles=sc.legend_elements()[0],
                   labels=["Q", "BC", "M", "X"])
    plt.savefig(f"plots/{hash_name}.png")


def train(context: Context, data: Data, reporter: Reporter):
    train, val, test = data.dataholders(context, *data.numpy_datasets(context))
    hash_name = util.hash_name(context)
    model = ConvModel(conv1_channels=context.ch_conv1,
                      conv2_channels=context.ch_conv2,
                      conv3_channels=context.ch_conv3,
                      l_hidden=context.l_hidden,
                      data_dropout=context.data_dropout,
                      layer_dropout=context.layer_dropout,
                      output_size=2 if context.binary else 4).to(Context.device)
    loss_weight = None
    if context.class_importance is not None:
        loss_weight = torch.Tensor(context.class_importance)
    criterion = nn.NLLLoss(weight=loss_weight).to(Context.device)
    algo = Algorithm(context,
                     model=model,
                     criterion=criterion,
                     optimizer=torch.optim.Adam(model.parameters(),
                                                lr=context.lr),
                     dataholder={"train": train,
                                 "val": val},
                     reporter=reporter,
                     verbose=True,
                     ablation=context.ablation)
    algo.train(early_stop=context.early_stop)
    test_loss, test_metric = algo.test(test)
    print(f"best val run: {algo.best_val_run_metric}")
    print(f"test run    : {test_metric}")
    torch.save(algo.best_model_wts,
               os.path.join(Context.model_dir, f"{hash_name}.ckpt"))
    if context.draw:
        draw(model.exp_last_layer(test[0].X).cpu().detach().numpy(),
             test[0].y.cpu().detach().numpy(),
             util.hash_name(context),
             context.binary)
    if reporter is not None:
        reporter.save_run_report(hash_name)
        reporter.split_row(context, algo.best_val_run_metric, test_metric)
        reporter.save_split_report(incremental=True)
    return algo.best_val_run_metric, test_metric


def cross_val(context, data, reporter):
    all_val_metric = Metric(binary=context.binary)
    all_test_metric = Metric(binary=context.binary)
    for test_part in range(1, 6):
        context.test_part = test_part
        best_val_run_metric, test_metric = train(context, data,
                                                 reporter=reporter)
        all_val_metric += best_val_run_metric
        all_test_metric += test_metric
    if reporter is not None:
        reporter.model_row(context,
                           all_val_metric=all_val_metric,
                           all_test_metric=all_test_metric)
        reporter.save_model_report(incremental=True)
    print(f"all test metric: {all_test_metric}")
    return all_val_metric, all_test_metric


def dataset_search(context, data, reporter):
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
        class_importances = [None,
                             [0.2, 0.2, 0.3, 0.3],
                             [0.1, 0.1, 0.4, 0.4],
                             [0.1, 0.1, 0.2, 0.6]]  # 4
    dropouts = [0.1 * i for i in range(0, 7, 2)]  # 4
    nan_modes = [0, None, "avg"]  # 3
    convs = [16, 32, 64, 128]  # 4
    hidden = [8, 16, 32, 64]  # 4
    normalizations = [Normalizer.scale, Normalizer.z_score]  # 2
    batch_sizes = [128, 1024, None]  # 3
    
    print(f"Searching through {context.n_random_search} items")
    for _ in range(context.n_random_search):
        if context.binary:
            split = [0] * 2
            split[0] = random.choice(bcq)
            split[1] = random.choice(mx)
        else:
            split = [0] * 4
            split[0] = random.choice(q)
            split[1] = random.choice(bc)
            split[2] = random.choice(m)
            split[3] = random.choice(x)
        training_mode = random.choice(training_modes)
        if training_mode == "n":
            context.train_n = split
            context.train_k = None
        else:
            context.train_k = split
            context.train_n = None
        
        context.ch_conv1 = random.choice(convs)
        context.ch_conv2 = random.choice(convs)
        context.ch_conv3 = random.choice(convs)
        context.l_hidden = random.choice(hidden)
        
        context.nan_mode = random.choice(nan_modes)
        context.data_dropout = random.choice(dropouts)
        context.layer_dropout = random.choice(dropouts)
        context.normalization_mode = random.choice(normalizations)
        context.class_importance = random.choice(class_importances)
        context.batch_size = random.choice(batch_sizes)
        print(context)
        cross_val(context, data, reporter)


def single_run(data, reporter, binary):
    binary_context = Context(binary=True,
                             train_n=[4000, 2200],
                             ch_conv1=64, ch_conv2=32, ch_conv3=0,
                             l_hidden=0,
                             nan_mode="avg",
                             batch_size=256,
                             normalization_mode="scale",
                             data_dropout=0.1,
                             layer_dropout=0.1,
                             val_p=0.5,
                             class_importance=[0.3, 0.7], stop=1000,
                             early_stop=40,
                             draw=True,
                             ablation=False)
    multi_context = Context(binary=False,
                            batch_size=256,
                            train_n=[2000, 2000, 400, 120],
                            ch_conv1=64, ch_conv2=32, ch_conv3=0,
                            l_hidden=32,
                            nan_mode="avg", normalization_mode="scale",
                            class_importance=[1, 1, 5, 15],
                            lr=0.01, data_dropout=0.2, layer_dropout=0.4,
                            val_p=0.5, stop=1000, early_stop=40, draw=True)
    cross_val(binary_context if binary else multi_context, data, reporter)


def main():
    # numpy.seterr(divide='ignore', invalid='ignore')
    numpy.set_printoptions(precision=2, formatter={'int': '{:5d}'.format,
                                                   'float': '{:7.2f}'.format})
    prog_args = util.arg_parse()
    start = time.time()
    Context.device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {Context.device}")
    print(f"cpu count: {os.cpu_count()}")
    Context.data_dir = "/home/arash/data/solar_flare"
    Context.tensorlog_dir = "tensorlog"
    Context.log_dir = "log"
    if not os.path.exists(Context.log_dir):
        os.makedirs(Context.log_dir)
    Context.model_dir = "models"
    if not os.path.exists(Context.model_dir):
        os.makedirs(Context.model_dir)
    Context.files_df_filename = "all_files.csv"
    Context.split_report_filename = f"split_report_{'binary' if prog_args.binary else 'multi'}.csv"
    Context.model_report_filename = f"model_report_{'binary' if prog_args.binary else 'multi'}.csv"
    Context.files_np_filename = "full_data_X_1_25.npy"
    context = Context(lr=0.01,
                      early_stop=40,
                      stop=400,
                      binary=prog_args.binary,
                      val_p=0.5,
                      n_random_search=prog_args.n_param_search,
                      run_times=prog_args.run_times)
    reporter = Reporter()
    data = Data()
    single_run(data, reporter, prog_args.binary)
    # dataset_search(context, data=data, reporter=reporter)
    run_time = int(time.time() - start)
    print(f"It took {run_time // 60:02d}:{run_time % 60:02d} to run program")


if __name__ == "__main__":
    main()
