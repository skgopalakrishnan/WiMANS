"""
[file]          ablstm.py
[description]   implement and evaluate WiFi-based model ABLSTM
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
from sklearn.preprocessing import StandardScaler
#
from train import train
from preset import preset
from preprocess import reduce_dimensionality

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
                 var_y_shape, 
                 hidden_dim = 256, 
                 drop = 0.5):
        #
        ##
        super(ABLSTM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_bilstm = torch.nn.LSTM(input_size = var_dim_input,
                                          hidden_size = hidden_dim,
                                          batch_first = True,
                                          bidirectional = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.layer_activation = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(2*hidden_dim, var_dim_output)
        #
        ##
        self.layer_softmax = torch.nn.Softmax(dim = -2)
        #
        ##
        self.layer_pooling = torch.nn.AvgPool1d(10, 10)
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_dropout = torch.nn.Dropout(drop)  

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

#
##
def run_ablstm(data_train_x,
               data_train_y,
               data_test_x,
               data_test_y,
               var_repeat = 10, 
               preprocessed = False):
    """
    [description]
    : run WiFi-based model ABLSTM
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    : preprocessed: bool, whether data is preprocessed
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device(preset["device"])
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    if not preprocessed: 
        data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
        data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
        #
        ## reduce dimensionality
        data = np.concatenate([data_train_x, data_test_x], axis = 0)
        data = reduce_dimensionality(data, new_chan = 10, new_seq_len = 50)
        data_train_x, data_test_x = data[:data_train_x.shape[0]], data[data_train_x.shape[0]:]
        #
        ## scaling
        scaler = StandardScaler()
        data_train_x = scaler.fit_transform(data_train_x.reshape(-1, data_train_x.shape[-1])).reshape(data_train_x.shape)
        data_test_x = scaler.transform(data_test_x.reshape(-1, data_test_x.shape[-1])).reshape(data_test_x.shape)
    else:
        assert len(np.shape(data_train_x)) == 3
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
    var_macs, var_params = get_model_complexity_info(ABLSTM(var_x_shape, var_y_shape), 
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
        model_ablstm = torch.compile(ABLSTM(var_x_shape, var_y_shape).to(device))
        #
        optimizer = torch.optim.Adam(model_ablstm.parameters(), 
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([6] * var_y_shape[-1]).to(device))
        #
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_ablstm, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_set = data_train_set,
                                data_test_set = data_test_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                model_type = "ABLSTM", 
                                run_ = var_r,
                                )
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_ablstm.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_ablstm(torch.from_numpy(data_test_x).to(device))
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
        ## Report
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