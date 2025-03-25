"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#
#
try:  # try relative imports in case of running as a module
    from .model import *
    from .preset import preset
    from .load_data import load_data_x, load_data_y, encode_data_y
    from .preprocess import reduce_dimensionality
except ImportError:  # for debugger
    from third_party.WiMANS.benchmark.wifi_csi.model import *
    from third_party.WiMANS.benchmark.wifi_csi.preset import preset
    from third_party.WiMANS.benchmark.wifi_csi.load_data import load_data_x, load_data_y, encode_data_y
    from third_party.WiMANS.benchmark.wifi_csi.preprocess import reduce_dimensionality
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
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    #
    return var_args.parse_args()

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    #
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    #
    ## load annotation file as labels
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = preset["data"]["environment"], 
                            var_wifi_band = preset["data"]["wifi_band"], 
                            var_num_users = preset["data"]["num_users"])
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    ## load CSI amplitude
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    ## preprocess data
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], -1)
    # data_x = reduce_dimensionality(data_x, new_chan=270, new_seq_len=100)
    ## sub-select features
    data_x = data_x[:, ::10, ::10]
    print("Data shape:", data_x.shape)
    #
    ## encode labels
    data_y = encode_data_y(data_pd_y, var_task)
    #
    ## a training set (70%), a validation set (15%), and a test set (15%)
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, 
                                                                            test_size = preset["train_test_split"], 
                                                                            shuffle = True, 
                                                                            random_state = 39)
    #
    ## split the test set into a validation set (50%) and a test set (50%)
    data_val_x, _, data_val_y, _ = train_test_split(data_test_x, data_test_y, 
                                                    test_size = 0.5, 
                                                    shuffle = True, 
                                                    random_state = 39)
    #
    ## Apply standard scaling to normalize data
    scaler = StandardScaler()
    data_train_x = scaler.fit_transform(data_train_x.reshape(-1, data_train_x.shape[-1])).reshape(data_train_x.shape)
    data_val_x = scaler.transform(data_val_x.reshape(-1, data_val_x.shape[-1])).reshape(data_val_x.shape)
    data_test_x = scaler.transform(data_test_x.reshape(-1, data_test_x.shape[-1])).reshape(data_test_x.shape)
    #
    ## select a WiFi-based model
    if var_model == "ST-RF": run_model = run_strf
    #
    elif var_model == "MLP": run_model = run_mlp
    #
    elif var_model == "LSTM": run_model = run_lstm
    #
    elif var_model == "CNN-1D": run_model = run_cnn_1d
    #
    elif var_model == "CNN-2D": run_model = run_cnn_2d
    #
    elif var_model == "CLSTM": run_model = run_cnn_lstm
    #
    elif var_model == "ABLSTM": run_model = run_ablstm
    #
    elif var_model == "THAT": run_model = run_that
    #
    ## run WiFi-based model
    result = run_model(data_train_x, data_train_y, 
                       data_val_x, data_val_y, var_repeat, preprocessed=True)
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print(result)
    #
    ## save results
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent = 4)

#
##
if __name__ == "__main__":
    #
    ##
    run()