import os
from copy import copy

import pandas as pd

import utils

text_normal = "\33[0m"
text_red = "\33[31m"
text_yellow = "\33[33m"
text_blue = "\33[34m"
text_negative = "\33[7m"


def save_report(loc, df, incremental):
    if incremental and os.path.exists(loc):
        pre_report_df = pd.read_csv(loc)
        pd.concat([pre_report_df, df]).round(
            decimals=4).to_csv(loc, index=False)
        df.drop(df.index, inplace=True)
    else:
        df.round(decimals=4).to_csv(loc, index=False)


def _print(loss, metric, postfix):
    if postfix is None: postfix = ""
    if metric.binary:
        print(f"\tloss {loss:.4f}, "
              f"{metric.tp + metric.tn:5d}/{metric.a:5d} | "
              f"\t"
              f"accuracy  {metric.accuracy * 100:5.2f} | "
              f"precision {metric.precision * 100:5.2f} | "
              f"recall    {metric.recall * 100:5.2f} | "
              f"f1        {metric.f1 * 100:5.2f} | \n"
              f"\t\t\t\t\t\t\t\t"
              f"TPR       {metric.tpr * 100:5.2f} | "
              f"TNR       {metric.tnr * 100:5.2f} | "
              f"TSS       {metric.tss * 100:5.2f} | \n"
              f"\t\t\t\t\t\t\t\t"
              f"HSS1    {metric.hss1 * 100:7.2f} | "
              f"HSS2    {metric.hss2 * 100:7.2f} | "
              f"GS      {metric.gs * 100:7.2f} | \n"
              f"\t\t\t\t\t\t\t\t"
              f"tp        {metric.tp:5d} , "
              f"fp        {metric.fp:5d} , "
              f"tn        {metric.tn:5d} , "
              f"fn        {metric.fn:5d}"
              f"{postfix}")
    else:

        print(f"\tloss {loss:.4f}, "
              f"{sum(metric.tp)}/{metric.a[0]} | "
              f"\t"
              f"accuracy  {metric.accuracy * 100} | "
              f"precision {metric.precision * 100} | "
              f"recall    {metric.recall * 100} | "
              f"f1        {metric.f1 * 100} | \n"
              f"\t\t\t\t\t\t\t\t"
              f"TPR       {metric.tpr * 100} | "
              f"TNR       {metric.tnr * 100} | "
              f"TSS       {metric.tss * 100} | \n"
              f"\t\t\t\t\t\t\t\t"
              f"HSS1      {metric.hss1 * 100} | "
              f"HSS2      {metric.hss2 * 100} | "
              f"GS        {metric.gs * 100} | \n"
              f"{postfix}")
        for i in metric.cm:
            print(f"\t\t\t\t\t\t\t\t{i}")


class BaselineReporter:
    def __init__(self):
        self.metric = None
        self.split_report_df = pd.DataFrame(
            columns=["run no.", "test_part", "train_k", "train_n", "nan_mode",
                     "normalization", "test_run"])
        self.model_report_df = pd.DataFrame(
            columns=["run no.", "train_k", "train_n", "nan_mode",
                     "normalization", "all_test_runs"])

    def model_row(self, args, all_test_metric):
        if args.verbose < 0: return
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            args.run_no,
            args.train_k,
            args.train_n,
            args.nan_mode,
            args.normalization_mode,
            all_test_metric]

    def split_row(self, args, test_metric):
        if args.verbose < 0: return
        self.split_report_df.loc[len(self.split_report_df.index)] = [
            args.run_no,
            args.test_part,
            args.train_k,
            args.train_n,
            args.nan_mode,
            args.normalization_mode,
            test_metric]

    def save_split_report(self, args, incremental=False):
        if args.verbose < 0: return
        loc = os.path.join(args.log_dir, args.split_report_filename)
        save_report(loc, self.model_report_df, incremental)

    def save_model_report(self, args, incremental=False):
        if args.verbose < 0: return
        loc = os.path.join(args.log_dir, args.model_report_filename)
        save_report(loc, self.model_report_df, incremental)


class Reporter:

    def __init__(self):
        self.loss = None
        self.metric = None
        self.report_df = None
        self.split_report_df = pd.DataFrame(
            columns=["id", "model id", "run no.", "val_part", "test_part",
                     "batch_size",
                     "train_k", "train_n", "ch_conv", "l_hidden", "dropout",
                     "nan_mode", "class_importance", "lr", "best_val_run",
                     "test_run"])
        self.model_report_df = pd.DataFrame(
            columns=["id", "run no.", "batch_size", "train_k", "train_n",
                     "ch_conv", "l_hidden", "dropout", "nan_mode",
                     "class_importance", "lr", "rand_seed", "np_seed", "torch_seed",
                     "all_best_val_runs",
                     "all_test_runs"])
        self.experiment = Reporter.ExperimentReporter()
        self.config = Reporter.ConfigReporter()
        self.run = Reporter.RunReporter()
        self.cross = Reporter.CrossReporter()
        self.epoch = Reporter.EpochReporter()

    def model_row(self, args, val_metric, test_metric):
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            utils.hash_model(args),
            args.run_no,
            args.batch_size,
            args.train_k,
            args.train_n,
            [args.ch_conv1, args.ch_conv2, args.ch_conv3],
            [args.l_hidden1, args.l_hidden2],
            [args.data_dropout, args.layer_dropout],
            args.nan_mode,
            args.class_importance,
            args.lr,
            args.rand_seed,
            args.np_seed,
            args.torch_seed,
            val_metric,
            test_metric]

    def split_row(self, args, best_val_metric, test_metric):
        self.split_report_df.loc[len(self.split_report_df.index)] = [
            utils.hash_name(args),
            utils.hash_model(args),
            args.run_no,
            args.val_part if args.val_part is not None else args.val_p,
            args.test_part,
            args.batch_size,
            args.train_k,
            args.train_n,
            [args.ch_conv1, args.ch_conv2, args.ch_conv3],
            [args.l_hidden1, args.l_hidden2],
            [args.data_dropout, args.layer_dropout],
            args.nan_mode,
            args.class_importance,
            args.lr,
            best_val_metric,
            test_metric]

    def save_split_report(self, args, incremental=False):
        if args.verbose < 0: return
        loc = os.path.join(args.log_dir, args.split_report_filename)
        save_report(loc, self.split_report_df, incremental)

    def save_model_report(self, args, incremental=False):
        if args.verbose < 0: return
        loc = os.path.join(args.log_dir, args.model_report_filename)
        save_report(loc, self.model_report_df, incremental)

    def update(self, args, loss, metric):
        self.loss = loss
        self.metric = copy(metric)

    class ExperimentReporter:

        @staticmethod
        def header(args):
            if args.verbose < 1: return
            print(f"-------- experiment {args.experiment} - "
                  f"searching {args.n_search} configs --------")

        @staticmethod
        def time(args, start, end):
            if args.verbose < 1: return
            duration = int(end - start)
            print(f"Running {args.experiment} with {args.n_search} configs took "
                  f"{duration // 60:02d}:{duration % 60:02d}.")

    class ConfigReporter:
        @staticmethod
        def print(args):
            if args.verbose < 2: return
            strat = "def"
            print(f"              Filter Size   Pool Size   Pool strat   Depth   Neurons\n"
                  f"Conv Block 1: {args.kernel_size[0]}             {args.pooling_size}           {strat}          {args.ch_conv1:3d}      -\n"
                  f"Conv Block 2: {args.kernel_size[1]}             {args.pooling_size}           {strat}          {args.ch_conv2:3d}      -\n"
                  f"Conv Block 3: {args.kernel_size[2]}             {args.pooling_size}           {strat}          {args.ch_conv3:3d}      -\n"
                  f"FCN Layer  1: -             -           -            -       {args.l_hidden1:3d}\n"
                  f"FCN Layer  1: -             -           -            -       {args.l_hidden2:3d}")

    class RunReporter:
        @staticmethod
        def val(args, metric):
            if args.verbose < 3: return
            print(f"Run no. {args.run_no}, val metric: {metric}")

        @staticmethod
        def test(args, metric):
            if args.verbose < 3: return
            print(f"Run no. {args.run_no}, test metric: {metric}")

    class CrossReporter:
        @staticmethod
        def time(args, start, end, n):
            if args.verbose < 3: return
            print(f"run no. {args.run_no}, test {args.test_part}:{n:5d}, duration {(end - start) * 1000:.1f} ms")

        @staticmethod
        def test(args, loss, metric):
            if args.verbose < 5: return
            print(text_negative)
            _print(loss, metric, "")
            print(text_normal)

        @staticmethod
        def best_val(args, metric):
            if args.verbose < 3: return
            print(f"run no. {args.run_no}, test {args.test_part}, best val run: {metric}")

        @staticmethod
        def best_test(args, metric):
            if args.verbose < 3: return
            print(f"run no. {args.run_no}, test {args.test_part}, test run    : {metric}") # todo: add test part

    class EpochReporter:
        @staticmethod
        def header(args, epoch, since_last_improve):
            if args.verbose < 5: return
            print(f"{text_red}"
                  f"run no. {args.run_no} | "
                  f"epoch {epoch + 1:4d} | "
                  f"early stop {since_last_improve:4d}: "
                  f"{text_normal}")

        @staticmethod
        def val(args, loss, metric, postfix=None):
            if args.verbose < 5: return
            print(text_yellow)
            _print(loss, metric, postfix)
            print(text_normal)

        @staticmethod
        def train(args, loss, metric, postfix=None):
            if args.verbose < 5: return
            print(text_blue)
            _print(loss, metric, postfix)
            print(text_normal)

        @staticmethod
        def time(args, start, end):
            if args.verbose < 5: return
            print(f"\t{(end - start) * 1000:.1f} ms")
