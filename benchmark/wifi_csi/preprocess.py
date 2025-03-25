"""
[file]          preprocess.py
[description]   preprocess WiFi CSI data
"""
#
##
import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

#
##
def mat_to_amp(data_mat):
    """
    [description]
    : calculate amplitude of raw WiFi CSI data
    [parameter]
    : data_mat: dict, raw WiFi CSI data from *.mat files
    [return]
    : data_csi_amp: numpy array, CSI amplitude
    """
    #
    ## 
    var_length = data_mat["trace"].shape[0]
    #
    data_csi_amp = [abs(data_mat["trace"][var_t][0][0][0][-1]) for var_t in range(var_length)]
    #
    data_csi_amp = np.array(data_csi_amp, dtype = np.float32)
    #
    return data_csi_amp

#
##
def extract_csi_amp(var_dir_mat, 
                    var_dir_amp):
    """
    [description]
    : read raw WiFi CSI files (*.mat), calcuate CSI amplitude, and save amplitude (*.npy)
    [parameter]
    : var_dir_mat: string, directory to read raw WiFi CSI files (*.mat)
    : var_dir_amp: string, directory to save WiFi CSI amplitude (*.npy)
    """
    #
    ##
    var_path_mat = os.listdir(var_dir_mat)
    #
    for var_c, var_path in enumerate(var_path_mat):
        #
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        #
        data_csi_amp = mat_to_amp(data_mat)
        #
        print(var_c, data_csi_amp.shape)
        #
        var_path_save = os.path.join(var_dir_amp, var_path.replace(".mat", ".npy"))
        #
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_amp)

#
##
def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--dir_mat", default = "dataset/wifi_csi/mat", type = str)
    var_args.add_argument("--dir_amp", default = "dataset/wifi_csi/amp", type = str)
    #
    return var_args.parse_args()

#
##
def reduce_dimensionality(data_x, new_chan=10, new_seq_len=100):
    """
    [description]
    : reduce dimensionality of WiFi CSI amplitudes
    [parameter]
    : data_x: numpy array, WiFi CSI data
    [return]
    : data_x_reduced: numpy array, simplified WiFi CSI data
    """
    #
    ## 
    cur_seq_len = data_x.shape[1]
    #
    ## downsample
    x = torch.tensor(data_x).float()
    x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
    x_downsampled = F.avg_pool1d(x, kernel_size=30, stride=int(cur_seq_len/new_seq_len))
    print("Post downsampling shape:", x_downsampled.shape)
    x = x_downsampled.permute(0, 2, 1)
    x = x.numpy()
    #
    ## PCA
    if new_chan >= x.shape[2]:
        print("No PCA needed")
        return x
    else:
        x = x.reshape(x.shape[0], -1)  # [batch_size, seq_len, channels] -> [batch_size, seq_len*channels]
        x_pca = PCA(n_components=new_chan*new_seq_len).fit_transform(x)
        print("Post PCA shape:", x_pca.shape)
        x = x_pca.reshape(x_pca.shape[0], new_seq_len, -1)
        #
        return x

#
##
if __name__ == "__main__":
    #
    ##
    var_args = parse_args()
    #
    extract_csi_amp(var_dir_mat = var_args.dir_mat, 
                    var_dir_amp = var_args.dir_amp)