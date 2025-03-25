"""
[file]          inference.py
[description]   collection of functions to generate inferences from WiFi-based models
"""
#
##
import os
import time
import numpy as np
import torch
import torch._dynamo
#
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score, classification_report
#
from .preset import preset
from .preprocess import reduce_dimensionality
from .utils import remove_prefix_from_state_dict
#
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536
#
##
def run_inference(model: Module,
            model_type: str,
            var_task: str,
            data_test_x: np.ndarray,
            data_test_y: np.ndarray,
            print_out: bool = False,
            need_reshape: bool = False,
            ) -> dict:
    """
    [description]
    : use WiFi-based model THAT to make predictions on a given test set and compute metrics
    [parameter]
    : model: Pytorch model to train
    : model_type: type of model (e.g., lstm, cnn, that)
    : var_task: str, task to run model on (e.g., "identity", "activity", "location")
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : print_out: bool, print out the predictions
    : need_reshape: bool, need to reshape the predictions before output
    [return]
    : result: dict, results of experiments
    """
    #
    #
    ## ============================================ Preprocess Data ============================================
    #
    ##
    # data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## ============================================ Load Model ============================================
    ## determine device to run model on
    device = torch.device(preset["device"])
    #
    ## load model weights and use to set up the model
    weights_fname = model_type.upper() + "_best_weight_" + var_task + ".pt"
    weights_fname = os.path.join(preset["path"]["model_wt"], weights_fname)
    weights_dict = torch.load(weights_fname, map_location=device)
    try:
        model.load_state_dict(weights_dict)
    except RuntimeError:  # remove prefix from state dict. _orig_mod is added to the key names
        model.load_state_dict(remove_prefix_from_state_dict(weights_dict))
    model.to(device)
    model.eval()
    #
    ## ========================================= Run Inference =========================================
    #
    ##
    result = {}
    result_accuracy = []
    result_time_test = []
    #
    var_time_1 = time.time()
    #
    with torch.no_grad():
        try:
            X = torch.from_numpy(data_test_x)
        except TypeError:
            X = data_test_x
        finally:
            X = X.to(device)
            predict_test_y = model(X)
    #
    predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
    predict_test_y = predict_test_y.detach().cpu().numpy()
    #
    # data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
    # predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
    data_test_y_c = data_test_y.reshape(data_test_y.shape[0], -1)
    predict_test_y_c = predict_test_y.reshape(data_test_y.shape[0], -1)
    #
    var_time_2 = time.time()
    #
    ## ========================================= Compute Metrics =================================
    #
    ## Accuracy
    result_acc = accuracy_score(data_test_y_c.astype(int),
                                predict_test_y_c.astype(int))
    #
    ## Report
    result_dict = classification_report(data_test_y.reshape(-1, data_test_y.shape[-1]),
                                        predict_test_y.reshape(-1, data_test_y.shape[-1]),
                                        digits=6,
                                        zero_division=0,
                                        output_dict=True)
    ## Print results
    if print_out:
        if need_reshape:
            predict_test_y = predict_test_y.reshape(data_test_y.shape[0], -1, data_test_y.shape[-1])
                # Print a header
            header = f"{'Sample':<8}{'Ground Truth':<25}{'Prediction':<25}{'Correct?':<10}"
            print(header)
            print("-" * len(header))
            # Loop over each sample
            for i in range(data_test_y.shape[0]):
                # Each sample is a 6x9 matrix
                truth_matrix = data_test_y[i]
                pred_matrix = predict_test_y[i]

                # Use np.unravel_index to find the indices of the maximum value.
                truth_user, truth_activity = np.unravel_index(np.argmax(truth_matrix, axis=None), truth_matrix.shape)
                pred_user, pred_activity = np.unravel_index(np.argmax(pred_matrix, axis=None), pred_matrix.shape)

                if truth_user == pred_user and truth_activity == pred_activity:
                    correct = "Yes"
                else:
                    correct = "No"
                
                # Format the result nicely
                truth_str = f"User {truth_user}, {var_task} {truth_activity}"
                pred_str = f"User {pred_user}, {var_task} {pred_activity}"
                correct_str = f"{correct}"
                print(f"{i:<8}{truth_str:<25}{pred_str:<25}{correct_str:<10}")

        else:
            print(f"  Ground truth: {data_test_y_c.argmax(axis=1)}")
            print(f"  Prediction:   {predict_test_y.argmax(axis=1)}")
    ## ========================================= Return results =========================================
    #
    result["accuracy"] = result_acc
    result["time_test"] = var_time_2 - var_time_1
    #
    return result
