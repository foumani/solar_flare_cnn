import itertools
import random

import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import os

import config
import utils
from algorithm import *
from conv_model import ConvModel
from data import Data
from preprocess import Normalizer
from reporter import Reporter


def draw(args, emb, y, hash_name, binary):
    tsne = TSNE(n_components=2 if binary else 4,
                method="barnes_hut" if binary else "exact")
    # plt.figure(figsize=(5, 3))

    qbc_idx = np.random.choice((y == 0).nonzero()[0], 1500, replace=False)
    mx_idx = np.random.choice((y == 1).nonzero()[0], 500, replace=False)
    tsne_results = tsne.fit_transform(np.append(emb[qbc_idx],
                                                emb[mx_idx],
                                                axis=0))
    y_emb = np.append(np.zeros(qbc_idx.shape[0]), np.ones(mx_idx.shape[0]))
    colors = ListedColormap(['green', 'red'])
    sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_emb,
                     cmap=colors, s=4)
    plt.legend(handles=sc.legend_elements()[0], labels=["QBC", "MX"])
    train_parts = [i for i in range(1, 6) if i not in [args.test_part]]
    # plt.title(f"Train parts {train_parts}, Test part {args.test_part}")
    plt.savefig(f"plots/{hash_name}.png")
    plt.show()


def train(args, data: Data, reporter: Reporter):
    train, val, test = data.dataholders(args, *data.numpy_datasets(args))
    model = ConvModel(args=args, output_size=2 if args.binary else 4).to(args.device)
    loss_weight = None
    if args.class_importance is not None:
        loss_weight = torch.Tensor(args.class_importance)
    criterion = nn.NLLLoss(weight=loss_weight).to(args.device)
    algo = Algorithm(args,
                     model=model,
                     criterion=criterion,
                     optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                     dataholder={"train": train, "val": val},
                     reporter=reporter)
    algo.train(early_stop=args.early_stop)
    test_loss, test_metric = algo.test(test)
    reporter.cross.best_val(args, algo.best_val_run_metric)
    reporter.cross.best_test(args, test_metric)
    if args.draw:
        draw(args,
             model.exp_last_layer(test[0].X).cpu().detach().numpy(),
             test[0].y.cpu().detach().numpy(),
             utils.hash_name(args),
             args.binary)
    reporter.split_row(args, algo.best_val_run_metric, test_metric)
    reporter.save_split_report(args, incremental=True)
    saliency = np.zeros([24, 60])
    if args.saliency:
        saliency = algo.my_saliency(test)
    return saliency, algo.best_val_run_metric, test_metric


def cross_val(args, data, reporter):
    all_val_metric = Metric(binary=args.binary)
    all_test_metric = Metric(binary=args.binary)
    saliency_all = np.zeros([24, 60])
    for test_part in range(1, 6):
        args.test_part = test_part
        saliency, best_val_run_metric, test_metric = train(args, data, reporter)
        saliency_all += saliency
        all_val_metric += best_val_run_metric
        all_test_metric += test_metric
    if reporter is not None:
        reporter.model_row(args, val_metric=all_val_metric, test_metric=all_test_metric)
        reporter.save_model_report(args, incremental=True)
    reporter.run.val(args, all_val_metric)
    reporter.run.test(args, all_test_metric)
    return saliency_all, all_val_metric, all_test_metric


def config_run(args, data, reporter):
    run_metrics = []
    utils.reset_seeds(args)
    for i in range(args.runs):
        args.run_no = i
        _, all_val_metric, all_test_metric = cross_val(args, data, reporter)
        run_metrics.append(all_test_metric)
    reporter.config_row(args, run_metrics)
    reporter.save_config_report(args, incremental=True)

def local_search(args, data, reporter):
    for pooling in config.poolings:
        config.optimal_model(args, binary=True)
        args.pooling_size = pooling[0]
        args.pooling_strat = pooling[1]
        config_run(args, data, reporter)

    for depth in config.depths:
        config.optimal_model(args, binary=True)
        args.depth = depth
        config_run(args, data, reporter)

    for hidden in config.hiddens:
        config.optimal_model(args, binary=True)
        args.hidden = hidden
        config_run(args, data, reporter)

    for filter_size in config.filter_sizes:
        config.optimal_model(args, binary=True)
        args.kernel_size = filter_size
        config_run(args, data, reporter)


def ablation(args, data, reporter):
    """
    Executes the optimal model with different configurations for including or excluding
    the last layer and the last convolution block.
    :param args:
    :param data:
    :param reporter:
    :return:
    """
    # last conv and last hidden gone
    config.optimal_model(args, binary=True)
    args.depth[2] = 0
    args.hidden[1] = 0
    config_run(args, data, reporter)

    # last conv gone
    config.optimal_model(args, binary=True)
    args.depth[2] = 0
    config_run(args, data, reporter)

    # last hidden gone
    config.optimal_model(args, binary=True)
    args.hidden[1] = 0
    config_run(args, data, reporter)

    # normal run
    config.optimal_model(args, binary=True)
    config_run(args, data, reporter)

def saliency_map(args, data, reporter):
    config.optimal_model(args, binary=True)
    utils.reset_seeds(args)

    args.run_no = 0
    args.saliency = True
    args.verbose = 5
    saliency, _, _ = cross_val(args, data, reporter)
    np.save("saliency.npy", saliency)
    # plt.figure(figsize=(8, 4))
    # plt.imshow(saliency, aspect='auto', cmap='hot')
    # plt.colorbar(label='Saliency')
    # plt.xlabel('Timesteps')
    # plt.ylabel('Features')
    # plt.title('Aggregated Saliency Map')
    # saliency = np.sum(saliency, axis=1)
    # plt.savefig(os.path.join(args.log_dir, "saliency_map.eps"))

    # plt.figure(figsize=(10, 5))
    # saliency_sum = np.sum(saliency, axis=1)
    # plt.plot(saliency_sum)

def search(args, data, reporter):
    for _ in range(args.n_search):
        args.train_n = random.choice(config.split_sizes)
        args.nan_mode = random.choice(config.nan_modes)
        args.normalization_mode = random.choice(config.normalizations)
        args.batch_size = 1024
        args.val_p = 0.5

        args.kernel_size = random.choice(config.filter_sizes)
        args.depth = random.choice(config.depths)
        args.pooling_size = random.choice([2, 3, 4])
        args.pooling_strat = random.choice(["mean", "max"])
        args.hidden = random.choice(config.hiddens)
        args.data_dropout = random.choice([0.2, 0.4, 0.6])
        args.layer_dropout = random.choice([0.1, 0.3, 0.5])
        args.class_importance = [0.4, 0.6]

        config_run(args, data, reporter)


def dataset_search(args, data, reporter):
    config.optimal_model(args, binary=True)

    dataset_grid = []
    training_modes = ["k", "n"]  # 2
    bcq = [i for i in range(400, 4001, 400)]  # 10
    mx = [i for i in range(200, 1401, 200)]  # 7
    dataset_grid.extend(list(itertools.product(bcq, mx)))

    nan_modes = [0, None, "avg"]  # 3

    normalizations = [Normalizer.scale, Normalizer.z_score]  # 2

    args.ablation = False

    reporter.experiment.header(args)

    for _ in range(args.n_search):
        split = [0] * 2
        split[0] = random.choice(bcq)
        split[1] = random.choice(mx)
        split = sorted(split, reverse=True)
        training_mode = random.choice(training_modes)
        if training_mode == "n":
            args.train_n = split
            args.train_k = None
        else:
            args.train_k = split
            args.train_n = None


        args.depth = random.choice(config.depths)
        args.hidden = random.choice(config.hiddens)
        args.nan_mode = random.choice(nan_modes)
        args.normalization_mode = random.choice(normalizations)

        args.run_no = 0  # Only run once for each parameter set.
        reporter.config.print(args)
        utils.reset_seeds(args)
        cross_val(args, data, reporter)


def single_run(args, data, reporter):
    # args.kernel_size = [7, 7, 5]
    # args.nan_mode = 0
    # args.batch_size = 256
    # args.normalization_mode = Normalizer.scale
    # args.data_dropout = 0.3
    # args.layer_dropout = 0.1
    # args.class_importance = [0.5, 0.5]
    # args.val_p = 0.3
    # args.runs = 100
    args.cache = True
    # args.rand_seed = 3764
    # args.np_seed = 7078
    utils.reset_seeds(args)
    for i in range(args.runs):
        args.run_no = i
        cross_val(args, data, reporter)


def single_serach(args, data, reporter):
    args.ch_conv1 = 32
    args.ch_conv2 = 64
    args.ch_conv3 = 128
    args.l_hidden1 = 64
    args.l_hidden2 = 32
    args.batch_size = 256
    args.data_dropout = 0.3
    args.layer_dropout = 0.1
    args.val_p = 0.4
    args.run_no = 1

    dataset_grid = []
    training_modes = ["k", "n"]  # 2

    bcq = [i for i in range(400, 4001, 400)]  # 10
    mx = [i for i in range(200, 1401, 200)]  # 7
    dataset_grid.extend(list(itertools.product(bcq, mx)))
    class_importances = [None,
                         [0.4, 0.6],
                         [0.3, 0.7],
                         [0.2, 0.8]]  # 4

    nan_modes = [0]  # 3
    normalizations = [Normalizer.scale, Normalizer.z_score]

    print(f"Searching through {args.n_search} items")
    for _ in range(5):
        split = [0] * 2
        split[0] = random.choice(bcq)
        split[1] = random.choice(mx)
        training_mode = random.choice(training_modes)
        if training_mode == "n":
            args.train_n = split
            args.train_k = None
        else:
            args.train_k = split
            args.train_n = None

        args.nan_mode = random.choice(nan_modes)
        args.normalization_mode = random.choice(normalizations)
        args.class_importance = random.choice(class_importances)
        cross_val(args, data, reporter)




def main():
    np.set_printoptions(precision=2, formatter={'int': '{:5d}'.format,
                                                'float': '{:7.2f}'.format})
    prog_args = utils.train_arg_parse()
    prog_args.cache = True
    utils.print_config(prog_args)
    reporter = Reporter()
    data = Data(prog_args)
    start = time.time()
    # single_serach(prog_args, data, reporter)
    single_run(prog_args, data, reporter)
    dataset_search(prog_args, data=data, reporter=reporter)

    reporter.experiment.time(prog_args, start, time.time())


if __name__ == "__main__":
    main()
