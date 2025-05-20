"""
[file]          reduce_dimensions.py
[description]   use an autoencoder model to reduce the dimensions of the CSI data
"""
############################################################################################################
## Import open-source libraries

import os
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info

# torch.autograd.set_detect_anomaly(True)

##########
## Local imports

from WiMANS.benchmark.wifi_csi.load_data import load_data_x, load_data_y, encode_data_y
from WiMANS.benchmark.wifi_csi.train import train

sys.path.append(os.path.abspath("/cs/academic/phd3/gopalak/Documents/Projects/Adversarial_CSI/src"))
from models.modules import LSTMAutoEncoder

############################################################################################################
## Set GPU device if available

if torch.cuda.is_available():            # Check if GPU is available
    device = "cuda"
elif torch.backends.mps.is_available():  # MacOS M-series
    device = "mps"
else:  # Neither
    device = "cpu"
print("Device:", device)

############################################################################################################
## Environmental variables

preset = {
    #
    ## define device for ML
    "device" : device,
    #
    ## define train-test-split
    "train_test_split": 0.3,                            # ratio of test set
    #
    ## define task
    "task": "activity",                                 # "identity", "activity", "location"
    #
    ## number of repeated experiments
    "repeat": 1,
    #
    ## path of data and model weights
    "path": {
        "data_x":   "/cs/academic/phd3/gopalak/Documents/Projects/Adversarial_CSI/data/WiMANS/dataset/wifi_csi/amp",                       # directory of CSI amplitude files
        "data_y":   "/cs/academic/phd3/gopalak/Documents/Projects/Adversarial_CSI/data/WiMANS/dataset/annotation.csv",                     # path of annotation file
        "save":     "/cs/academic/phd3/gopalak/Documents/Projects/Adversarial_CSI/data/WiMANS/dataset/result.json",                        # path to save results
        "model_wt": "/cs/academic/phd3/gopalak/Documents/Projects/Adversarial_CSI/third_party/WiMANS/src/WiMANS/benchmark/saved_models",   # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["1"],                               # select number(s) of users
        "wifi_band": ["2.4", "5"],                        # select WiFi band(s)
        "environment": ["classroom"],                     # select environment(s)
        "length": 3000,                                   # default length of CSI
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 1e-4,                                     # learning rate
        "epoch": 200,                                   # number of epochs
        "batch_size": 10,                                # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "early_stop": 1000,                             # early stopping patience
    },
    "autoenc_model":  {
        "input_size" : 270, # input size: number of channels
        "hidden_size_enc" : 32, # hidden size of encoder for dimensionality reduction
        "hidden_size_dec" : 32, # hidden size of encoder for dimensionality reduction
        "seq_len": 300,  # length of the CSI data sequences
        "attention": True  # whether to use attention
        }
    }

##########
## set seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


############################################################################################################
## Function definitions

def read_dataset():
    
    ## load annotation file as labels
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_wifi_band = preset["data"]["wifi_band"], 
                            var_num_users = preset["data"]["num_users"])

    var_label_list = data_pd_y["label"].to_list()
    del data_pd_y  # clear up memory

    ## load CSI amplitude
    data_x = load_data_x(preset["path"]["data_x"], var_label_list, debug=True)

    ## preprocess data
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], -1)
    
    print("Sub-sampling data_x...")
    data_x = data_x[:, ::10, :]
    
    print("Data shape:", data_x.shape)
    return data_x

##########

def preprocess_dataset():
    
    ## a training set (70%), a validation set (15%), and a test set (15%)
    data_train_x, data_test_x = train_test_split(data_x, test_size = preset["train_test_split"], 
                                                shuffle = True, 
                                                random_state = 39)

    ## split the test set into a validation set (50%) and a test set (50%)
    data_val_x, data_test_x = train_test_split(data_test_x, test_size = 0.5,
                                               shuffle = True,
                                               random_state = 39)

    ## Apply standard scaling to normalize data
    scaler = StandardScaler()
    data_train_x = scaler.fit_transform(data_train_x.reshape(-1, data_train_x.shape[-1])).reshape(data_train_x.shape)
    data_val_x = scaler.transform(data_val_x.reshape(-1, data_val_x.shape[-1])).reshape(data_val_x.shape)
    data_test_x = scaler.transform(data_test_x.reshape(-1, data_test_x.shape[-1])).reshape(data_test_x.shape)
    
    return data_train_x, data_val_x, data_test_x

##########

def train_model(Model, data_train_x, data_val_x, data_test_x):
    
    ## shape for model
    var_x_shape = data_train_x[0].shape
    input_size = preset["autoenc_model"]["input_size"]
    hidden_size_enc = preset["autoenc_model"]["hidden_size_enc"]
    hidden_size_dec = preset["autoenc_model"]["hidden_size_dec"]
    output_size = preset["autoenc_model"]["input_size"]
    seq_len = preset["autoenc_model"]["seq_len"]
    attention = preset["autoenc_model"]["attention"]
    
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_x))
    data_val_set = TensorDataset(torch.from_numpy(data_val_x), torch.from_numpy(data_val_x))    
    
    ## ========================================= Train & Evaluate =========================================

    result = {}

    try:
        var_macs, var_params = get_model_complexity_info(Model(input_size, hidden_size_enc, hidden_size_dec, 
                                                            output_size, seq_len, attention), 
                                                            var_x_shape, as_strings = False)
        print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    except TypeError:
        pass

    for var_r in range(1):

        print("Repeat", var_r)
            
        model = torch.compile(Model(input_size, hidden_size_enc, hidden_size_dec, 
                                    output_size, seq_len, attention).to(device))
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr = preset["nn"]["lr"],
                                    weight_decay = 0)
        loss = torch.nn.MSELoss().to(device)

        # ---------------------------------------- Train -----------------------------------------
        
        var_best_weight = train(model = model, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_set = data_train_set,
                                data_test_set = data_val_set,
                                var_threshold = None,  # for reconstruction
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                model_type=Model.__name__,
                                run_ = var_r,
                                save_path = preset["path"]["model_wt"]
                                )
        ## ---------------------------------------- Test ------------------------------------------

        model.load_state_dict(var_best_weight)

        with torch.no_grad():
            predict_test_y = model(torch.from_numpy(data_test_x).to(device))
            var_loss_test = loss(predict_test_y, torch.from_numpy(data_test_x).to(device))
            print("Test loss:", var_loss_test.item())

        result["repeat_" + str(var_r)] = var_loss_test.item()

    return result

##########

def reduce_dimensionality(Model, saved_weight_path, data_test_x):
    
    saved_weight_path = os.path.join(saved_weight_path, "LSTMAutoEncoder_best_weight_run-1.pt")
    print("Loading saved weight from:", saved_weight_path)    
    saved_weight = torch.load(saved_weight_path)
    
    model = torch.compile(Model(preset["autoenc_model"]["input_size"],
                                preset["autoenc_model"]["hidden_size_enc"],
                                preset["autoenc_model"]["hidden_size_dec"],
                                preset["autoenc_model"]["input_size"],
                                preset["autoenc_model"]["seq_len"], 
                                preset["autoenc_model"]["attention"])).to(device)
    model.load_state_dict(saved_weight)
    
    with torch.no_grad():
        predict_test_y = model.reduce_dims(torch.from_numpy(data_test_x).to(device))
    
    print(predict_test_y.shape)

    return predict_test_y.cpu().numpy()

############################################################################################################

if __name__ == "__main__":
    
    data_x = read_dataset()
    data_train_x, data_val_x, data_test_x = preprocess_dataset()
    result = train_model(LSTMAutoEncoder, data_train_x, data_val_x, data_test_x)
    reduce_result = reduce_dimensionality(LSTMAutoEncoder, preset["path"]["model_wt"], data_test_x)
    print("Result:", result)

############################################################################################################
