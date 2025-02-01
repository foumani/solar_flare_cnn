import os
from copy import copy

import pandas as pd

import util

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
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            args.run_no,
            args.train_k,
            args.train_n,
            args.nan_mode,
            args.normalization_mode,
            all_test_metric]
    
    def split_row(self, args, test_metric):
        self.split_report_df.loc[len(self.split_report_df.index)] = [
            args.run_no,
            args.test_part,
            args.train_k,
            args.train_n,
            args.nan_mode,
            args.normalization_mode,
            test_metric]
    
    def save_split_report(self, args, incremental=False):
        loc = os.path.join(args.log_dir, args.split_report_filename)
        save_report(loc, self.model_report_df, incremental)
    
    def save_model_report(self, args, incremental=False):
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
                     "class_importance", "lr", "rand_seed", "np_seed", "torch_seed", "all_best_val_runs",
                     "all_test_runs"])
    
    def model_row(self, args, val_metric, test_metric):
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            util.hash_model(args),
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
            util.hash_name(args),
            util.hash_model(args),
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
        loc = os.path.join(args.log_dir, args.split_report_filename)
        save_report(loc, self.split_report_df, incremental)
    
    def save_model_report(self, args, incremental=False):
        loc = os.path.join(args.log_dir, args.model_report_filename)
        save_report(loc, self.model_report_df, incremental)
    
    def update(self, loss, metric):
        self.metric = copy(metric)
        self.loss = loss
    
    def print(self, postfix=""):
        if self.metric.binary:
            print(f"\tloss {self.loss:.4f}, "
                  f"{self.metric.tp + self.metric.tn:5d}/{self.metric.a:5d} | "
                  f"\t"
                  f"accuracy  {self.metric.accuracy * 100:5.2f} | "
                  f"precision {self.metric.precision * 100:5.2f} | "
                  f"recall    {self.metric.recall * 100:5.2f} | "
                  f"f1        {self.metric.f1 * 100:5.2f} | \n"
                  f"\t\t\t\t\t\t\t\t"
                  f"TPR       {self.metric.tpr * 100:5.2f} | "
                  f"TNR       {self.metric.tnr * 100:5.2f} | "
                  f"TSS       {self.metric.tss * 100:5.2f} | \n"
                  f"\t\t\t\t\t\t\t\t"
                  f"HSS1    {self.metric.hss1 * 100:7.2f} | "
                  f"HSS2    {self.metric.hss2 * 100:7.2f} | "
                  f"GS      {self.metric.gs * 100:7.2f} | \n"
                  f"\t\t\t\t\t\t\t\t"
                  f"tp        {self.metric.tp:5d} , "
                  f"fp        {self.metric.fp:5d} , "
                  f"tn        {self.metric.tn:5d} , "
                  f"fn        {self.metric.fn:5d}"
                  f"{'| ' if postfix != '' else ''}{postfix}")
        else:
            
            print(f"\tloss {self.loss:.4f}, "
                  f"{sum(self.metric.tp)}/{self.metric.a[0]} | "
                  f"\t"
                  f"accuracy  {self.metric.accuracy * 100} | "
                  f"precision {self.metric.precision * 100} | "
                  f"recall    {self.metric.recall * 100} | "
                  f"f1        {self.metric.f1 * 100} | \n"
                  f"\t\t\t\t\t\t\t\t"
                  f"TPR       {self.metric.tpr * 100} | "
                  f"TNR       {self.metric.tnr * 100} | "
                  f"TSS       {self.metric.tss * 100} | \n"
                  f"\t\t\t\t\t\t\t\t"
                  f"HSS1      {self.metric.hss1 * 100} | "
                  f"HSS2      {self.metric.hss2 * 100} | "
                  f"GS        {self.metric.gs * 100} | \n"
                  f"{'| ' if postfix != '' else ''}{postfix}")
            for i in self.metric.cm:
                print(f"\t\t\t\t\t\t\t\t{i}")
