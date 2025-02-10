import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):

    def __init__(self, args, output_size):
        super(ConvModel, self).__init__()
        self.args = args
        self.output_size = output_size
        self.conv1 = nn.Conv1d(in_channels=24,
                               out_channels=self.args.depth[0],
                               kernel_size=args.kernel_size[0],
                               bias=True)  # [ n_batch x conv1_ch x time-series ]
        self.pool1 = nn.MaxPool1d(kernel_size=args.pooling_size,
                                  stride=2)  # [ n_batch x conv1_ch x time-series ]
        self.conv2 = nn.Conv1d(in_channels=self.args.depth[0],
                               out_channels=self.args.depth[1],
                               kernel_size=args.kernel_size[1],
                               bias=True)  # [ n_batch x conv2_ch x time-series ]
        self.pool2 = nn.MaxPool1d(kernel_size=args.pooling_size,
                                  stride=2)  # [ n_batch x conv2_ch x time-series ]

        if self.args.depth[2] > 0:
            self.conv3 = nn.Conv1d(in_channels=self.args.depth[1],
                                   out_channels=self.args.depth[2],
                                   kernel_size=args.kernel_size[2],
                                   bias=True)  # [ n_batch x conv3_ch x time-series ]
            self.pool3 = nn.MaxPool1d(kernel_size=args.pooling_size,
                                      stride=2)  # [ n_batch x conv3_ch x time-series ]
        # else:
        #     self.ablation_pool = nn.MaxPool1d(kernel_size=20,
        #                                       stride=20)  # [ n_batch x conv2_ch * 1 ]

        self.linear1 = nn.LazyLinear(out_features=self.args.hidden[0])
        self.feature_size = self.args.hidden[0]
        if self.args.hidden[1] > 0:
            self.linear2 = nn.Linear(in_features=self.args.hidden[0],
                                     out_features=self.args.hidden[1])
            self.feature_size = self.args.hidden[1]

        self.l_out = nn.Linear(in_features=self.feature_size,
                               out_features=output_size)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        if self.args.depth[2] > 0:
            torch.nn.init.xavier_uniform_(self.conv3.weight)
        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        if self.args.hidden[1] > 0:
            torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.l_out.weight)
        self.batch_norm1 = nn.BatchNorm1d(self.args.depth[0])
        self.batch_norm2 = nn.BatchNorm1d(self.args.depth[1])

    def forward(self, X):
        X = F.dropout(X, p=self.args.data_dropout)

        X = self.pool1(F.leaky_relu(self.conv1(X)))
        X = F.dropout(X, p=self.args.layer_dropout)

        X = self.pool2(F.leaky_relu(self.conv2(X)))
        X = F.dropout(X, p=self.args.layer_dropout)

        if self.args.depth[2] > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
            X = F.dropout(X, p=self.args.layer_dropout)

        X = X.reshape(X.shape[0], -1)

        X = F.leaky_relu(self.linear1(X))
        X = F.dropout(X, p=self.args.layer_dropout)

        if self.args.hidden[1] > 0:
            X = F.leaky_relu(self.linear2(X))
            X = F.dropout(X, p=self.args.layer_dropout)

        X = F.log_softmax(self.l_out(X), dim=1)

        return X

    def ablation_study(self, X):
        X = F.dropout(X, p=self.data_dropout)
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        if self.training:
            X = F.dropout(X, p=self.layer_dropout)
        if self.depth[2] > 0:
            X = self.pool2(F.leaky_relu(self.conv2(X)))
            if self.training:
                X = F.dropout(X, p=self.layer_dropout)
            X = self.pool3(F.leaky_relu(self.conv3(X)))
            X = F.dropout(X, p=self.layer_dropout)
        else:
            X = self.ablation_pool(F.leaky_relu(self.conv2(X)))
            X = F.dropout(X, p=self.layer_dropout)
        X = X.reshape(-1, self.output_size)
        X = F.log_softmax(X, dim=1)
        return X

    def convolution_layer(self, X):
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        X = self.pool2(F.leaky_relu(self.conv2(X)))
        if self.depth[2] > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
        X = X.reshape(-1, self.conv_size)
        return X

    def exp_last_layer(self, X):
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        X = self.pool2(F.leaky_relu(self.conv2(X)))
        if self.depth[2] > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
        X = X.reshape(-1, self.conv_size)
        X = F.leaky_relu(self.linear1(X))
        if self.hidden[1] > 0:
            X = F.leaky_relu(self.linear2(X))
        return X
