import itertools
import os
import random

import torch.nn as nn

import util
from algorithm import *
from context import Context
from conv_model import ConvModel
from data import Data
from preprocess import Normalizer
from reporter import Reporter


def train(context: Context, data: Data):
    train, _, test = data.dataholders(context, *data.numpy_datasets(context),
                                        test=True)
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
                     dataholder={"train": train},
                     verbose=False)
    algo.train(early_stop=context.early_stop)
    test_loss, test_metric = algo.test(test)
    print(f"test run    : {test_metric}")
    return test_metric
