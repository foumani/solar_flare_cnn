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
from orion.client import create_experiment
import ast
import json


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
    train, val, test = data.dataholders(args, *data.numpy_datasets(args, args.run_no))
    if args.ordering is not None:
        train, val, test = data.select_indices(train, val, test, args)
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
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
    args.test_part = 5
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
    utils.add_results(args, run_metrics)
    # 🔁 Return the mean TSS across runs
    tss_scores = [m.tss for m in run_metrics]
    avg_tss = sum(tss_scores) / len(tss_scores)
    print(f"Average TSS of {args.runs} runs: {avg_tss:.4f}")
    return {"loss": -avg_tss}  # BOHB minimizes loss, so use -tss


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
    return saliency
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
    search_rng = random.Random(args.seed)
    for _ in range(args.n_search):
        args.train_n = search_rng.choice(config.split_sizes)
        # args.nan_mode = search_rng.choice(config.nan_modes)
        args.nan_mode = "local_avg"
        args.normalization_mode = search_rng.choice(config.normalizations)
        # args.batch_size = 1024
        args.val_p = 0.5
        args.kernel_size = search_rng.choice(config.filter_sizes)
        args.depth = search_rng.choice(config.depths)
        args.pooling_size = search_rng.choice([2, 3, 4])
        args.pooling_strat = search_rng.choice(["mean", "max"])
        args.hidden = search_rng.choice(config.hiddens)
        args.data_dropout = 0.0
        args.layer_dropout = search_rng.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        args.class_importance = search_rng.choice(config.importance)
        config_run(args, data, reporter)


def bohb_search(args, data, reporter):
    def stringify_choices(choices):
        return "choices(" + ", ".join(repr(json.dumps(c)) for c in choices) + ")"

    experiment = create_experiment(
        name="debug_bohb_test",
        space={
            "budget": "fidelity(1, 200, 1)",
            "lr": "loguniform(1e-5, 1e-1)"
        },
        algorithm={
            "bohb": {
                "seed": 42,
            }
        },
        max_trials=10,
    )

    space = {
        "kernel_size": stringify_choices(config.filter_sizes),
        "depth": stringify_choices(config.depths),
        "hidden": stringify_choices(config.hiddens),
        "pooling": stringify_choices(config.poolings),
        "train_n": stringify_choices(config.split_sizes),
        "normalization_mode": f"choices{tuple(config.normalizations)}",
        "layer_dropout": "choices(0.1, 0.2, 0.3, 0.4, 0.5)",
        "class_importance": stringify_choices(config.importance),
        # "epochs": "fidelity(1, 200, 1)",  # 👈 must match exactly
    }

    print(space)

    experiment = create_experiment(
        name="bohb_config_search_full",
        space=space,
        algorithm={"bohb": {"seed": 42}, "budgets": {
                "min": 1,
                "max": 200
            }},
        max_trials=args.n_search,
    )

    for trial in experiment.suggested_trials:
        args.train_n = json.loads(trial.params["train_n"])
        args.kernel_size = json.loads(trial.params["kernel_size"])
        args.depth = json.loads(trial.params["depth"])
        args.hidden = json.loads(trial.params["hidden"])
        args.pooling_size, args.pooling_strat = json.loads(trial.params["pooling"])
        args.class_importance = json.loads(trial.params["class_importance"])
        args.nan_mode = "local_avg"
        args.normalization_mode = trial.params["normalization_mode"]
        args.val_p = 0.5
        args.depth = ast.literal_eval(trial.params["depth"])
        args.data_dropout = 0.0
        args.layer_dropout = trial.params["layer_dropout"]

        args = config.optimal_model(args, binary=True)
        result = config_run(args, data, reporter)
        trial.observe(result["loss"])

    # Optional: best result
    best = experiment.best_trial
    print("Best:", best.params, "Score:", best.objective)


if __name__ == "__main__":
    main()
