import time
from copy import deepcopy

import numpy as np
import torch

from reporter import Reporter
from utils import Metric


class Algorithm:

    def __init__(self, args, model, criterion, optimizer, dataholder,
                 reporter: Reporter):
        super(Algorithm, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataholder = dataholder
        self.reporter = reporter
        self.best_model_wts = None
        self.best_val_run_metric = Metric(binary=self.args.binary)
        self.ablation = args.ablation

    def train(self, early_stop=5):
        epoch, early_stop_cnt = 0, 0
        train_metric = Metric()
        best_f1 = 0.0
        while early_stop_cnt <= early_stop and epoch <= self.args.stop:
            epoch_start_time = time.time()
            self.reporter.epoch.header(self.args, epoch, early_stop_cnt)
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                dataholder = self.dataholder[phase]
                epoch_loss, metric, _ = self.run_epoch(dataholder)
                if self.reporter is not None:
                    self.reporter.update(self.args, epoch_loss, metric)
                if phase == "val":
                    if best_f1 <= np.average(train_metric.f1):
                        early_stop_cnt = -1
                        best_f1 = np.average(train_metric.f1)
                        self.best_val_run_metric = deepcopy(metric)
                        self.best_model_wts = deepcopy(self.model.state_dict())
                    else:
                        early_stop_cnt += 1
                    self.reporter.epoch.val(self.args, epoch_loss, metric,
                                            postfix="improved" if early_stop_cnt == -1 else None)
                else:
                    train_metric = deepcopy(metric)
                    self.reporter.epoch.train(self.args, epoch_loss, metric)
                if early_stop_cnt == -1:
                    early_stop_cnt += 1
            epoch += 1
            self.reporter.epoch.time(self.args, epoch_start_time, time.time())

    def test(self, dataholder):
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        epoch_start_time = time.time()
        avg_loss, metric, l_dataloader = self.run_epoch(dataholder)
        self.reporter.cross.time(self.args, epoch_start_time, time.time(), n=l_dataloader)
        self.reporter.update(self.args, avg_loss, metric)
        self.reporter.cross.test(self.args, avg_loss, metric)
        return avg_loss, metric

    def emb(self, dataholder):
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        emb = np.empty(shape=(0, self.model.conv3_channels))
        y_emb = np.empty((0,))
        for i, (X, y) in enumerate(dataholder):
            X_emb = self.model.exp_last_layer(X).detach().cpu().numpy()
            emb = np.append(emb, X_emb, axis=0)
            y_emb = np.append(y_emb, y.cpu().numpy().copy())
        return emb, y_emb

    def run_epoch(self, dataloader):
        running_loss, metric = 0, Metric(binary=self.args.binary)
        l_points = 0
        l_dataloader = 0
        for i, (X, y) in enumerate(dataloader):
            if X.get_device() != self.args.device:
                X = X.to(self.args.device)
            if y.get_device() != self.args.device:
                y = y.to(self.args.device)
            if self.ablation:
                output = self.model.ablation_study(X)
            else:
                output = self.model(X)
            loss = self.criterion(output, y)
            if self.model.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            with torch.no_grad():
                running_loss += loss.item() * X.size(0)
                _, y_pred = torch.max(output, dim=1)
                metric += Metric(y.cpu().numpy(), y_pred.cpu().numpy(),
                                 self.args.binary)
                l_points += len(y)
                l_dataloader += len(y)
        return running_loss / l_points, metric, l_dataloader
