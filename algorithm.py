import time
from copy import deepcopy

import numpy
import numpy as np
import torch

import reporter
from reporter import Reporter
from util import Metric


class Algorithm:
    
    def __init__(self, args, model, criterion, optimizer, dataholder,
                 reporter: Reporter, verbose=True):
        super(Algorithm, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataholder = dataholder
        self.verbose = verbose
        self.reporter = reporter
        self.best_model_wts = None
        self.best_val_run_metric = Metric(binary=self.args.binary)
        if self.reporter is None:
            self.verbose = False
        self.ablation = args.ablation
    
    def train(self, early_stop=5):
        epoch, early_stop_cnt = 0, 0
        train_metric = Metric()
        best_f1 = 0.0
        while early_stop_cnt <= early_stop and epoch <= self.args.stop:
            epoch_start_time = time.time()
            if self.verbose:
                print(reporter.text_red +
                      f"run no. {self.args.run_no} | "
                      f"epoch {epoch + 1:4d} | "
                      f"early stop {early_stop_cnt:4d}: "
                      + reporter.text_normal)
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                dataholder = self.dataholder[phase]
                epoch_loss, metric, _ = self.run_epoch(dataholder)
                if self.reporter is not None:
                    self.reporter.update(epoch_loss, metric)
                if phase == "val":
                    if self.verbose:
                        print(reporter.text_blue)
                    if best_f1 <= np.average(train_metric.f1):
                        early_stop_cnt = -1
                        best_f1 = np.average(train_metric.f1)
                        self.best_val_run_metric = deepcopy(metric)
                        self.best_model_wts = deepcopy(self.model.state_dict())
                    else:
                        early_stop_cnt += 1
                else:
                    train_metric = deepcopy(metric)
                    if self.verbose:
                        print(reporter.text_yellow)
                if self.reporter is not None:
                    self.reporter.print(
                        postfix="improved" if early_stop_cnt == -1 else "")
                if phase == "val":
                    if self.verbose:
                        print(reporter.text_normal)
                if early_stop_cnt == -1:
                    early_stop_cnt += 1
            epoch += 1
            if self.verbose:
                print(f"\t{(time.time() - epoch_start_time) * 1000:.1f} ms")
    
    def test(self, dataholder):
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        epoch_start_time = time.time()
        avg_loss, metric, l_dataloader = self.run_epoch(dataholder)
        print(f"\t{(time.time() - epoch_start_time) * 1000:.1f} ms for the test with {l_dataloader} elements")
        if self.reporter is not None:
            self.reporter.update(avg_loss, metric)
        if self.verbose:
            print(reporter.text_negative)
            self.reporter.print()
            print(reporter.text_normal)
        return avg_loss, metric
    
    def emb(self, dataholder):
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        emb = numpy.empty(shape=(0, self.model.conv3_channels))
        y_emb = numpy.empty((0,))
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
