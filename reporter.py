import os
from copy import copy

import pandas as pd

import util
from context import Context

text_normal = "\33[0m"
text_red = "\33[31m"
text_yellow = "\33[33m"
text_blue = "\33[34m"
text_negative = "\33[7m"


class BaselineReporter:
    def __init__(self, split_report_filename, model_report_filename):
        self.split_report_filename = split_report_filename
        self.model_report_filename = model_report_filename
        self.metric = None
        self.split_report_df = pd.DataFrame(
            columns=["run no.", "test_part", "train_k", "train_n", "nan_mode",
                     "normalization", "test_run"])
        self.model_report_df = pd.DataFrame(
            columns=["run no.", "train_k", "train_n", "nan_mode",
                     "normalization", "all_test_runs"])
    
    def model_row(self, context: Context, all_test_metric):
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            context.run_no,
            context.train_k,
            context.train_n,
            context.nan_mode,
            context.normalization_mode,
            all_test_metric]
    
    def split_row(self, context, test_metric):
        self.split_report_df.loc[len(self.split_report_df.index)] = [
            context.run_no,
            context.test_part,
            context.train_k,
            context.train_n,
            context.nan_mode,
            context.normalization_mode,
            test_metric]
    
    def save_split_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, self.split_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.split_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.split_report_df.drop(self.split_report_df.index,
                                      inplace=True)
        else:
            self.split_report_df.round(decimals=4).to_csv(loc, index=False)
    
    def save_model_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, self.model_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.model_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.model_report_df.drop(self.model_report_df.index, inplace=True)
        else:
            self.model_report_df.round(decimals=4).to_csv(loc, index=False)


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
                     "class_importance", "lr", "all_best_val_runs",
                     "all_test_runs"])
    
    def new_run(self):
        self.report_df = pd.DataFrame(columns=[
            "epoch", "avg_loss", "tp", "fp", "tn", "fn", "p", "n", "a",
            "accuracy", "precision", "recall", "f1", "tss", "hss1", "hss2", "gs"
        ])
    
    def model_row(self, context: Context, all_val_metric, all_test_metric):
        self.model_report_df.loc[len(self.model_report_df.index)] = [
            util.hash_model(context),
            context.run_no,
            context.batch_size,
            context.train_k,
            context.train_n,
            [context.ch_conv1, context.ch_conv2, context.ch_conv3],
            [context.l_hidden],
            [context.data_dropout, context.layer_dropout],
            context.nan_mode,
            context.class_importance,
            context.lr,
            all_val_metric,
            all_test_metric]
    
    def split_row(self, context, best_val_metric, test_metric):
        self.split_report_df.loc[len(self.split_report_df.index)] = [
            util.hash_name(context),
            util.hash_model(context),
            context.run_no,
            context.val_part if context.val_part is not None else context.val_p,
            context.test_part,
            context.batch_size,
            context.train_k,
            context.train_n,
            [context.ch_conv1, context.ch_conv2, context.ch_conv3],
            [context.l_hidden],
            [context.data_dropout, context.layer_dropout],
            context.nan_mode,
            context.class_importance,
            context.lr,
            best_val_metric,
            test_metric]
    
    def save_split_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, Context.split_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.split_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.split_report_df.drop(self.split_report_df.index,
                                      inplace=True)
        else:
            self.split_report_df.round(decimals=4).to_csv(loc, index=False)
    
    def save_model_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, Context.model_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.model_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.model_report_df.drop(self.model_report_df.index, inplace=True)
        else:
            self.model_report_df.round(decimals=4).to_csv(loc, index=False)
    
    def run_row(self, epoch):
        self.report_df.loc[len(self.report_df.index)] = [epoch,
                                                         self.loss,
                                                         self.metric.tp,
                                                         self.metric.fp,
                                                         self.metric.tn,
                                                         self.metric.fn,
                                                         self.metric.p,
                                                         self.metric.n,
                                                         self.metric.a,
                                                         self.metric.accuracy,
                                                         self.metric.precision,
                                                         self.metric.recall,
                                                         self.metric.f1,
                                                         self.metric.tss,
                                                         self.metric.hss1,
                                                         self.metric.hss2,
                                                         self.metric.gs]
    
    def update(self, loss, metric):
        self.metric = copy(metric)
        self.loss = loss
    
    def save_run_report(self, hash_name, incremental=False):
        return
        loc = os.path.join(Context.log_dir, f"{hash_name}-report.csv")
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.report_df]).to_csv(loc, index=False)
        else:
            self.report_df.to_csv(loc, index=False)
    
    def save_split_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, Context.split_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.split_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.split_report_df.drop(self.split_report_df.index, inplace=True)
        else:
            self.split_report_df.round(decimals=4).to_csv(loc, index=False)
    
    def save_model_report(self, incremental=False):
        loc = os.path.join(Context.log_dir, Context.model_report_filename)
        if incremental and os.path.exists(loc):
            pre_report_df = pd.read_csv(loc)
            pd.concat([pre_report_df, self.model_report_df]).round(
                decimals=4).to_csv(loc, index=False)
            self.model_report_df.drop(self.model_report_df.index, inplace=True)
        else:
            self.model_report_df.round(decimals=4).to_csv(loc, index=False)
    
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
