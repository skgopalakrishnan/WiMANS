"""
[file]          lstm.py
[description]   define architecture of LSTM model
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
from sklearn.preprocessing import StandardScaler
#
from ..train import train
from ..preset import preset
from ..preprocess import reduce_dimensionality

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- LSTM --------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class LSTMM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(LSTMM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        #
        self.layer_pooling = torch.nn.AvgPool1d(10, 10)
        #
        self.layer_lstm = torch.nn.LSTM(input_size = var_dim_input,
                                        hidden_size = 512, 
                                        batch_first = True)
        #
        self.layer_linear = torch.nn.Linear(512, var_dim_output)

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
        var_t = self.layer_pooling(var_t)
        var_t = torch.permute(var_t, (0, 2, 1))
        #
        var_t, _ = self.layer_lstm(var_t)
        #
        var_t = var_t[:, -1, :]
        #
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
