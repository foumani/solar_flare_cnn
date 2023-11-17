import itertools
import os
import random

import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

import util
from algorithm import *
from conv_model import ConvModel
from data import Data
from preprocess import Normalizer
from reporter import Reporter


def draw(args, emb, y, hash_name, binary):
    tsne = TSNE(n_components=2 if binary else 4,
                method="barnes_hut" if binary else "exact")
    # plt.figure(figsize=(5, 3))
    
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
    train_parts = [i for i in range(1, 6) if i not in [args.test_part]]
    plt.title(f"Train parts {train_parts}, Test part {args.test_part}")
    plt.savefig(f"plots/{hash_name}.png")
    plt.show()


def train(args, data: Data, reporter: Reporter):
    train, val, test = data.dataholders(args, *data.numpy_datasets(args))
    model = ConvModel(conv1_channels=args.ch_conv1,
                      conv2_channels=args.ch_conv2,
                      conv3_channels=args.ch_conv3,
                      l_hidden=args.l_hidden,
                      data_dropout=args.data_dropout,
                      layer_dropout=args.layer_dropout,
                      output_size=2 if args.binary else 4).to(args.device)
    loss_weight = None
    if args.class_importance is not None:
        loss_weight = torch.Tensor(args.class_importance)
    criterion = nn.NLLLoss(weight=loss_weight).to(args.device)
    algo = Algorithm(args,
                     model=model,
                     criterion=criterion,
                     optimizer=torch.optim.Adam(model.parameters(),
                                                lr=args.lr),
                     dataholder={"train": train,
                                 "val": val},
                     reporter=reporter,
                     verbose=True)
    algo.train(early_stop=args.early_stop)
    test_loss, test_metric = algo.test(test)
    print(f"best val run: {algo.best_val_run_metric}")
    print(f"test run    : {test_metric}")
    if args.draw:
        draw(args,
             model.exp_last_layer(test[0].X).cpu().detach().numpy(),
             test[0].y.cpu().detach().numpy(),
             util.hash_name(args),
             args.binary)
    if reporter is not None:
        reporter.split_row(args, algo.best_val_run_metric, test_metric)
        reporter.save_split_report(args, incremental=True)
    return algo.best_val_run_metric, test_metric


def cross_val(args, data, reporter):
    all_val_metric = Metric(binary=args.binary)
    all_test_metric = Metric(binary=args.binary)
    for test_part in range(1, 6):
        args.test_part = test_part
        best_val_run_metric, test_metric = train(args, data, reporter)
        all_val_metric += best_val_run_metric
        all_test_metric += test_metric
    if reporter is not None:
        reporter.model_row(args,
                           val_metric=all_val_metric,
                           test_metric=all_test_metric)
        reporter.save_model_report(args, incremental=True)
    print(f"all test metric: {all_test_metric}")
    return all_val_metric, all_test_metric


def dataset_search(args, data, reporter):
    dataset_grid = []
    training_modes = ["k", "n"]  # 2
    if args.binary:
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
    hidden = [0, 8, 16, 32, 64]  # 5
    normalizations = [Normalizer.scale, Normalizer.z_score]  # 2
    batch_sizes = [128, 1024, None]  # 3
    
    print(f"Searching through {args.n_random_search} items")
    for _ in range(args.n_random_search):
        if args.binary:
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
            args.train_n = split
            args.train_k = None
        else:
            args.train_k = split
            args.train_n = None
        
        args.ch_conv1 = random.choice(convs)
        args.ch_conv2 = random.choice(convs)
        args.ch_conv3 = random.choice(convs + [0])
        args.l_hidden = random.choice(hidden)
        
        args.nan_mode = random.choice(nan_modes)
        args.data_dropout = random.choice(dropouts)
        args.layer_dropout = random.choice(dropouts)
        args.normalization_mode = random.choice(normalizations)
        args.class_importance = random.choice(class_importances)
        args.batch_size = random.choice(batch_sizes)
        args.run_no = 0  # Only run once for each parameter set.
        cross_val(args, data, reporter)


def single_run(args, data, reporter):
    if args.binary:
        args.train_n = [4000, 2200]
        args.ch_conv1 = 64
        args.ch_conv2 = 32
        args.ch_conv3 = 0
        args.l_hidden = 0
        args.nan_mode = "avg"
        args.batch_size = 256
        args.normalization_mode = "scale"
        args.data_dropout = 0.1
        args.layer_dropout = 0.1
        args.val_p = 0.5
        args.class_importance = [0.3, 0.7]
    else:
        args.batch_size = 256
        args.train_n = [2000, 2000, 400, 120],
        args.ch_conv1 = 64
        args.ch_conv2 = 32
        args.ch_conv3 = 0
        args.l_hidden = 32,
        args.nan_mode = "avg"
        args.normalization_mode = "scale"
        args.class_importance = [1, 1, 5, 15]
        args.lr = 0.01
        args.data_dropout = 0.2
        args.layer_dropout = 0.4
        args.val_p = 0.5
    
    cross_val(args, data, reporter)


def main():
    numpy.set_printoptions(precision=2, formatter={'int': '{:5d}'.format,
                                                   'float': '{:7.2f}'.format})
    prog_args = util.train_arg_parse()
    start = time.time()
    print(f"cpu count: {os.cpu_count()}")
    print(prog_args)
    print(f"device: {prog_args.device}")
    print(f"data dir: {prog_args.data_dir}")
    print(f"binary: {prog_args.binary}")
    print(f"csv database: {prog_args.files_df_filename}")
    print(f"mem instances: {prog_args.files_np_filename}")
    print(f"val p: {prog_args.val_p}")
    print(f"early stop: {prog_args.early_stop}")
    reporter = Reporter()
    data = Data(prog_args)
    # single_run(prog_args, data, reporter)
    dataset_search(prog_args, data=data, reporter=reporter)
    run_time = int(time.time() - start)
    print(f"It took {run_time // 60:02d}:{run_time % 60:02d} to run program")


if __name__ == "__main__":
    main()
