"""
[file]          that.py
[description]   define architecture of THAT model
                https://github.com/windofshadow/THAT
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
# from third_party.WiMANS.benchmark.wifi_csi.train import train

# from WiMANS.benchmark.wifi_csi.preset import preset
from ..preset import preset

#
##
## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Gaussian Encoding ------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Gaussian_Position(torch.nn.Module):
    #
    ##
    def __init__(self, 
                 var_dim_feature,
                 var_dim_time, 
                 var_num_gaussian = 10):
        #
        ##
        super(Gaussian_Position, self).__init__()
        #
        ## var_embedding: shape (var_dim_k, var_dim_feature)
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype = torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad = True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        #
        ## var_position: shape (var_dim_time, var_dim_k)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad = False)
        #
        ## var_mu: shape (1, var_dim_k)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time/var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad = True)
        #
        ## var_sigma: shape (1, var_dim_k)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad = True)

    #
    ##
    def calculate_pdf(self,
                      var_position, 
                      var_mu, 
                      var_sigma):
        #
        ##
        var_pdf = var_position - var_mu                 # (position-mu)
        #
        var_pdf = - var_pdf * var_pdf                   # -(position-mu)^2
        #
        var_pdf = var_pdf / var_sigma / var_sigma / 2   # -(position-mu)^2 / (2*sigma^2)
        #
        var_pdf = var_pdf - torch.log(var_sigma)        # -(position-mu)^2 / (2*sigma^2) - log(sigma)
        #
        return var_pdf

    #
    ##
    def forward(self, 
                var_input):
        

        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)
        
        var_pdf = torch.softmax(var_pdf, dim = -1)
        #
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        #
        # print(var_input.shape, var_position_encoding.shape)
        var_output = var_input + var_position_encoding.unsqueeze(0)
        #
        return var_output

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Encoder ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Encoder(torch.nn.Module):
    #
    ##
    def __init__(self, 
                 var_dim_feature, 
                 var_num_head = 10,
                 var_size_cnn = [1, 3, 5]):
        #
        ##
        super(Encoder, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps = 1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature, 
                                                           var_num_head,
                                                           batch_first = True)
        #
        self.layer_dropout_0 = torch.nn.Dropout(0.1)
        #
        ##
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, 1e-6)
        #
        layer_cnn = []
        #
        for var_size in var_size_cnn:
            #
            layer = torch.nn.Sequential(torch.nn.Conv1d(var_dim_feature,
                                                        var_dim_feature,
                                                        var_size, 
                                                        padding = "same"),
                                        torch.nn.BatchNorm1d(var_dim_feature),
                                        torch.nn.Dropout(0.1),
                                        torch.nn.LeakyReLU())
            layer_cnn.append(layer)
        #
        self.layer_cnn = torch.nn.ModuleList(layer_cnn)
        #
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

    #
    ##
    def forward(self, 
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_norm_0(var_t)
        #
        var_t, _ = self.layer_attention(var_t, var_t, var_t)

        var_t = self.layer_dropout_0(var_t)
        #
        var_t = var_t + var_input
        #
        ## 
        var_s = self.layer_norm_1(var_t)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim = 0)
        #
        var_s = torch.sum(var_c, dim = 0) / len(self.layer_cnn)
        #
        var_s = self.layer_dropout_1(var_s)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_output = var_s + var_t
        #
        return var_output
    
#
##
## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- THAT -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class THAT(torch.nn.Module):
    #
    ##
    def __init__(self, 
                 var_x_shape, 
                 var_y_shape, 
                 var_num_head_enc = 2):
        #
        ##
        super(THAT, self).__init__()
        #
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        self.layer_left_pooling = torch.nn.AvgPool1d(kernel_size = 20, stride = 20)
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, var_dim_time // 20)
        #
        var_num_left = 4
        var_dim_left = var_dim_feature
        self.layer_left_encoder = torch.nn.ModuleList([Encoder(var_dim_feature = var_dim_left,
                                                               var_num_head = var_num_head_enc,
                                                               var_size_cnn = [1, 3, 5])
                                                               for _ in range(var_num_left)])
        #
        self.layer_left_norm = torch.nn.LayerNorm(var_dim_left, eps = 1e-6)
        #
        self.layer_left_cnn_0 =  torch.nn.Conv1d(in_channels = var_dim_left,
                                                 out_channels = 128,
                                                 kernel_size = 8)
        
        self.layer_left_cnn_1 =  torch.nn.Conv1d(in_channels = var_dim_left,
                                                 out_channels = 128,
                                                 kernel_size = 8)
        #
        self.layer_left_dropout = torch.nn.Dropout(0.5)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        self.layer_right_pooling = torch.nn.AvgPool1d(kernel_size = 20, stride = 20)
        #
        var_num_right = 1 
        var_dim_right = var_dim_time // 20
        self.layer_right_encoder = torch.nn.ModuleList([Encoder(var_dim_feature = var_dim_right,
                                                                var_num_head = var_num_head_enc,
                                                                var_size_cnn = [1, 2, 3])
                                                                for _ in range(var_num_right)])
        #
        self.layer_right_norm = torch.nn.LayerNorm(var_dim_right, eps = 1e-6)
        #
        self.layer_right_cnn_0 =  torch.nn.Conv1d(in_channels = var_dim_right,
                                                  out_channels = 16,
                                                  kernel_size = 2)
        
        self.layer_right_cnn_1 =  torch.nn.Conv1d(in_channels = var_dim_right,
                                                  out_channels = 16,
                                                  kernel_size = 4)
        #
        self.layer_right_dropout = torch.nn.Dropout(0.5)
        #
        ##
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)
    
    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input   # shape (batch_size, time_steps, features)
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        var_left = torch.permute(var_t, (0, 2, 1))
        var_left = self.layer_left_pooling(var_left)
        var_left = torch.permute(var_left, (0, 2, 1))
        #
        var_left = self.layer_left_gaussian(var_left)
        #
        for layer in self.layer_left_encoder: var_left = layer(var_left)
        #
        var_left = self.layer_left_norm(var_left)
        #
        var_left = torch.permute(var_left, (0, 2, 1))
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        #
        var_left_0 = torch.sum(var_left_0, dim = -1)
        var_left_1 = torch.sum(var_left_1, dim = -1)
        #
        var_left = torch.concat([var_left_0, var_left_1], dim = -1)
        var_left = self.layer_left_dropout(var_left)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        var_right = torch.permute(var_t, (0, 2, 1)) # shape (batch_size, features, time_steps)
        var_right = self.layer_right_pooling(var_right)
        #
        for layer in self.layer_right_encoder: var_right = layer(var_right)
        #
        var_right = self.layer_right_norm(var_right)
        #
        var_right = torch.permute(var_right, (0, 2, 1))
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        #
        var_right_0 = torch.sum(var_right_0, dim = -1)
        var_right_1 = torch.sum(var_right_1, dim = -1)
        #
        var_right = torch.concat([var_right_0, var_right_1], dim = -1)
        var_right = self.layer_right_dropout(var_right)
        #
        ## concatenate
        var_t = torch.concat([var_left, var_right], dim = -1)
        #
        var_output = self.layer_output(var_t)
        #
        ##
        return var_output
