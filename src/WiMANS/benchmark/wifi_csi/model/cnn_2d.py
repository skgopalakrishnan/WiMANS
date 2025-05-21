"""
[file]          cnn_2d.py
[description]   define architecture of CNN-2D model
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
## --------------------------------------- CNN-2D ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class CNN_2D(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(CNN_2D, self).__init__()
        #
        ##
        var_dim_input = var_x_shape
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm_0 = torch.nn.BatchNorm2d(1)
        self.layer_norm_1 = torch.nn.BatchNorm2d(32)
        self.layer_norm_2 = torch.nn.BatchNorm2d(64)
        self.layer_norm_3 = torch.nn.BatchNorm2d(128)
        #
        self.layer_cnn_2d_0 = torch.nn.Conv2d(in_channels = 1, 
                                              out_channels = 32,
                                              kernel_size = (27, 27),
                                              stride = (7, 7))
        #
        self.layer_cnn_2d_1 = torch.nn.Conv2d(in_channels = 32, 
                                              out_channels = 64,
                                              kernel_size = (15, 15),
                                              stride = (3, 3))
        #
        self.layer_cnn_2d_2 = torch.nn.Conv2d(in_channels = 64, 
                                              out_channels = 128,
                                              kernel_size = (7, 7),
                                              stride = (1, 1))
        #
        self.layer_linear = torch.nn.Linear(128, var_dim_output)
        #
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        self.layer_dropout = torch.nn.Dropout(0.2)
        #
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_2d_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_linear.weight)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.unsqueeze(var_t, dim = 1)
        #
        var_t = self.layer_norm_0(var_t)
        var_t = self.layer_cnn_2d_0(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_1(var_t)
        var_t = self.layer_cnn_2d_1(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_2(var_t)
        var_t = self.layer_cnn_2d_2(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_norm_3(var_t)
        var_t = torch.mean(var_t, dim = (-2, -1))
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
