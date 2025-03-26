"""
[file]          mlp.py
[description]   define architecture of MLP model
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
## ---------------------------------------- MLP --------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class MLP(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(MLP, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        #
        self.layer_0 = torch.nn.Linear(var_dim_input, 256)
        self.layer_1 = torch.nn.Linear(256, 128)
        self.layer_2 = torch.nn.Linear(128, var_dim_output)
        #
        self.layer_relu = torch.nn.ReLU()
        self.layer_dropout = torch.nn.Dropout(0.1)
        #
        torch.nn.init.xavier_uniform_(self.layer_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_norm(var_t)
        #
        var_t = self.layer_0(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_1(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_2(var_t)
        var_t = self.layer_dropout(var_t)
        #
        var_output = var_t
        #
        return var_output
