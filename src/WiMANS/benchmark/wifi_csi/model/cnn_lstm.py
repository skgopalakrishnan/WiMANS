"""
[file]          cnn_lstm.py
[description]   implement and evaluate WiFi-based model CLSTM
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
## --------------------------------------- CLSTM -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class CNN_LSTM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(CNN_LSTM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_norm_0 = torch.nn.BatchNorm1d(64)
        self.layer_norm_1 = torch.nn.BatchNorm1d(128)
        self.layer_norm_2 = torch.nn.BatchNorm1d(256)
        #
        self.layer_cnn_1d_0 = torch.nn.Conv1d(in_channels = var_dim_input, 
                                              out_channels = 64,            # 64
                                              kernel_size = 128,            # 128
                                              stride = 8)                   # 8
        #
        self.layer_cnn_1d_1 = torch.nn.Conv1d(in_channels = 64,             
                                              out_channels = 128,           # 128
                                              kernel_size = 64,             # 64
                                              stride = 4)                   # 4
        #
        self.layer_cnn_1d_2 = torch.nn.Conv1d(in_channels = 128, 
                                              out_channels = 256,           # 256
                                              kernel_size = 32,             # 32
                                              stride = 2)                   # 2
        #
        self.layer_lstm = torch.nn.LSTM(input_size = 256,
                                        hidden_size = 512,                  # 512
                                        batch_first = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(512, var_dim_output)
        #
        ##
        self.layer_dropout = torch.nn.Dropout(0.5)      # 0.5
        #
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
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
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_0(var_t)
        #
        var_t = self.layer_cnn_1d_1(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_1(var_t)
        #
        var_t = self.layer_cnn_1d_2(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_norm_2(var_t)
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        #
        var_t, _ = self.layer_lstm(var_t)
        #
        var_t = var_t[:, -1, :]
        #
        var_t = self.layer_dropout(var_t)
        #
        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output
     
#
##
def run_cnn_lstm(data_train_x,
                 data_train_y,
                 data_test_x,
                 data_test_y,
                 var_repeat = 10):
    """
    [description]
    : run WiFi-based model CLSTM
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape, data_train_y[0].reshape(-1).shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(CNN_LSTM(var_x_shape, var_y_shape), 
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_cnn_lstm = torch.compile(CNN_LSTM(var_x_shape, var_y_shape).to(device))
        #
        optimizer = torch.optim.Adam(model_cnn_lstm.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([8] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_cnn_lstm, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_set = data_train_set,
                                data_test_set = data_test_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device)
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_cnn_lstm.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_cnn_lstm(torch.from_numpy(data_test_x).to(device))
        #
        predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##
        data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
        predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
        #
        ## Accuracy
        result_acc = accuracy_score(data_test_y_c.astype(int), 
                                    predict_test_y_c.astype(int))
        #
        ##
        result_dict = classification_report(data_test_y_c, 
                                            predict_test_y_c, 
                                            digits = 6, 
                                            zero_division = 0, 
                                            output_dict = True)
        #
        result["repeat_" + str(var_r)] = result_dict
        #
        result_accuracy.append(result_acc)
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        #
        print("repeat_" + str(var_r), result_accuracy)
        print(result)
    #
    ##
    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    result["complexity"] = {"parameter": var_params, "flops": var_macs * 2}
    #
    return result