import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
    
    def __init__(self, conv1_channels, conv2_channels, conv3_channels,
                 l_hidden1, l_hidden2, data_dropout, layer_dropout, output_size):
        super(ConvModel, self).__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.conv3_channels = conv3_channels
        self.l_hidden1 = l_hidden1
        self.l_hidden2 = l_hidden2
        self.data_dropout = data_dropout
        self.layer_dropout = layer_dropout
        self.output_size = output_size
        self.conv1 = nn.Conv1d(in_channels=24,
                               out_channels=self.conv1_channels,
                               kernel_size=7,
                               bias=True)  # [ n_batch x conv1_ch x 54 ]
        self.pool1 = nn.MaxPool1d(kernel_size=4,
                                  stride=2)  # [ n_batch x conv1_ch x 24 ]
        self.conv2 = nn.Conv1d(in_channels=self.conv1_channels,
                               out_channels=self.conv2_channels,
                               kernel_size=7,
                               bias=True)  # [ n_batch x conv2_ch x 20 ]
        self.pool2 = nn.MaxPool1d(kernel_size=5,
                                  stride=2 if self.conv3_channels > 0 else 5)  # [ n_batch x conv2_ch x 8 or 4 ]
        
        self.conv_size = self.conv2_channels * 4
        if self.conv3_channels != 0:
            self.conv3 = nn.Conv1d(in_channels=self.conv2_channels,
                                   out_channels=self.conv3_channels,
                                   kernel_size=5,
                                   bias=True)  # [ n_batch x conv3_ch x 4 ]
            self.pool3 = nn.MaxPool1d(kernel_size=4,
                                      stride=2)  # [ n_batch x conv3_ch x 1 ]
            self.conv_size = self.conv3_channels
        else:
            self.ablation_pool = nn.MaxPool1d(kernel_size=20,
                                              stride=20)  # [ n_batch x conv2_ch * 1 ]

        self.linear1 = nn.Linear(in_features=self.conv_size,
                                 out_features=self.l_hidden1)
        self.feature_size = self.l_hidden1
        if self.l_hidden2 != 0:
            self.linear2 = nn.Linear(in_features=self.l_hidden1,
                                    out_features=self.l_hidden2)
            self.feature_size = self.l_hidden2
        
        self.l_out = nn.Linear(in_features=self.feature_size,
                               out_features=output_size)
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        if self.conv3_channels > 0:
            torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        if self.l_hidden2 > 0:
            torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.l_out.weight)
        self.batch_norm1 = nn.BatchNorm1d(conv1_channels)
        self.batch_norm2 = nn.BatchNorm1d(conv2_channels)
    
    def forward(self, X):
        X = F.dropout(X, p=self.data_dropout)
        
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        X = F.dropout(X, p=self.layer_dropout)
        
        X = self.pool2(F.leaky_relu(self.conv2(X)))
        X = F.dropout(X, p=self.layer_dropout)
        
        if self.conv3_channels > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
            X = F.dropout(X, p=self.layer_dropout)
        
        X = X.reshape(-1, self.conv_size)

        X = F.leaky_relu(self.linear1(X))
        X = F.dropout(X, p=self.layer_dropout)

        if self.l_hidden2 > 0:
            X = F.leaky_relu(self.linear2(X))
            X = F.dropout(X, p=self.layer_dropout)

        X = F.log_softmax(self.l_out(X), dim=1)
        
        return X
    
    def ablation_study(self, X):
        X = F.dropout(X, p=self.data_dropout)
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        if self.training:
            X = F.dropout(X, p=self.layer_dropout)
        if self.conv3_channels > 0:
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
        if self.conv3_channels > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
        X = X.reshape(-1, self.conv_size)
        return X
    
    def exp_last_layer(self, X):
        X = self.pool1(F.leaky_relu(self.conv1(X)))
        X = self.pool2(F.leaky_relu(self.conv2(X)))
        if self.conv3_channels > 0:
            X = self.pool3(F.leaky_relu(self.conv3(X)))
        X = X.reshape(-1, self.conv_size)
        X = F.leaky_relu(self.linear1(X))
        if self.l_hidden2 > 0:
            X = F.leaky_relu(self.linear2(X))
        return X
