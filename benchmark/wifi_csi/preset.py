"""
[file]          preset.py
[description]   default settings of WiFi-based models
"""
#
##
import torch
if torch.cuda.is_available():            # Check if GPU is available
    device = "cuda"
elif torch.backends.mps.is_available():  # MacOS M-series
    device = "mps"
else:  # Neither
    device = "cpu"
#
##
preset = {
    #
    ## define device for ML
    "device" : device,
    #
    ## define model
    "model": "ABLSTM",                                  # "ST-RF", "MLP", "LSTM", "CNN-1D", "CNN-2D", "CLSTM", "ABLSTM", "THAT"
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
        "data_x": "./data/WiMANS/dataset/wifi_csi/amp",                     # directory of CSI amplitude files
        "data_y": "./data/WiMANS/dataset/annotation.csv",                   # path of annotation file
        "save": "./data/WiMANS/dataset/result.json",                        # path to save results
        "model_wt": "./third_party/WiMANS/benchmark/wifi_csi/saved_models", # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        # "num_users": ["0", "1", "2", "3", "4", "5"],    # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "num_users": ["2"],                               # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["2.4", "5"],                        # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["classroom"],                     # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,                                   # default length of CSI (3 seconds capture at 1000 Hz)
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 1e-3,                                     # learning rate
        "epoch": 200,                                   # number of epochs
        "batch_size": 64,                               # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "early_stop": 1000,                               # early stopping patience
    },
    #
    ## encoding of activities and locations
    "encoding": {
        "activity": {                                   # encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "location": {                                   # encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
}
