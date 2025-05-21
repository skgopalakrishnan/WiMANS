"""
[file]          cnn_1d.py
[description]   define architecture of CNN-1D model
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
## --------------------------------------- CNN-1D ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class CNN_1D(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(CNN_1D, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        #
        ##
        self.layer_cnn_1d_0 = torch.nn.Conv1d(in_channels = var_dim_input, 
                                              out_channels = 128,
                                              kernel_size = 29,
                                              stride = 13)
        #
        self.layer_cnn_1d_1 = torch.nn.Conv1d(in_channels = 128, 
                                              out_channels = 256,
                                              kernel_size = 15,
                                              stride = 7)
        #
        self.layer_cnn_1d_2 = torch.nn.Conv1d(in_channels = 256, 
                                              out_channels = 512,
                                              kernel_size = 3,
                                              stride = 1)
        #
        ##
        self.layer_linear = torch.nn.Linear(512, var_dim_output)
        #
        ##
        self.layer_dropout = torch.nn.Dropout(0.2)
        #
        self.layer_relu = torch.nn.ReLU()
        #
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
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_cnn_1d_1(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_cnn_1d_2(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = torch.mean(var_t, dim = -1)
        
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
