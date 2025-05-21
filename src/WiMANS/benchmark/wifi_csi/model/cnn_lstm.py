"""
[file]          cnn_lstm.py
[description]   define architecture of CLSTM model
"""
#
##
import time
import torch
import numpy as np
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from ..train import train
from ..preset import preset

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- CLSTM -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class CNN_LSTM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(CNN_LSTM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_norm_0 = torch.nn.BatchNorm1d(64)
        self.layer_norm_1 = torch.nn.BatchNorm1d(128)
        self.layer_norm_2 = torch.nn.BatchNorm1d(256)
        #
        self.layer_cnn_1d_0 = torch.nn.Conv1d(in_channels = var_dim_input, 
                                              out_channels = 64,            # 64
                                              kernel_size = 128,            # 128
                                              stride = 8)                   # 8
        #
        self.layer_cnn_1d_1 = torch.nn.Conv1d(in_channels = 64,             
                                              out_channels = 128,           # 128
                                              kernel_size = 64,             # 64
                                              stride = 4)                   # 4
        #
        self.layer_cnn_1d_2 = torch.nn.Conv1d(in_channels = 128, 
                                              out_channels = 256,           # 256
                                              kernel_size = 32,             # 32
                                              stride = 2)                   # 2
        #
        self.layer_lstm = torch.nn.LSTM(input_size = 256,
                                        hidden_size = 512,                  # 512
                                        batch_first = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(512, var_dim_output)
        #
        ##
        self.layer_dropout = torch.nn.Dropout(0.5)      # 0.5
        #
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_linear.weight)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        var_t = self.layer_norm(var_t)
        #
        var_t = self.layer_cnn_1d_0(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_0(var_t)
        #
        var_t = self.layer_cnn_1d_1(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_1(var_t)
        #
        var_t = self.layer_cnn_1d_2(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_2(var_t)
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        #
        var_t, _ = self.layer_lstm(var_t)
        #
        var_t = var_t[:, -1, :]
        #
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
