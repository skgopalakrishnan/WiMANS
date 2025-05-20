"""
[file]          ablstm.py
[description]   define architecture of ABLSTM model
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
## --------------------------------------- ABLSTM ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class ABLSTM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(ABLSTM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_bilstm = torch.nn.LSTM(input_size = var_dim_input,
                                          hidden_size = 512,
                                          batch_first = True,
                                          bidirectional = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(2*512, 2*512)
        self.layer_activation = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(2*512, var_dim_output)
        #
        ##
        self.layer_softmax = torch.nn.Softmax(dim = -2)
        #
        ##
        self.layer_pooling = torch.nn.AvgPool1d(8, 8)
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_dropout = torch.nn.Dropout(0.6)  

        torch.nn.init.xavier_uniform_(self.layer_linear.weight)
        torch.nn.init.xavier_uniform_(self.layer_output.weight)
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
        var_h, _ = self.layer_bilstm(var_t)

        var_s = self.layer_linear(var_h)
        var_s = self.layer_activation(var_s)

        var_a = self.layer_softmax(var_s)

        var_t = var_h * var_a

        #
        var_t = torch.sum(var_t, dim = -2)

        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_output(var_t)

        var_output = var_t
        #
        return var_output
