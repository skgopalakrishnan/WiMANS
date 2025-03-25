"""
[file]          train.py
[description]   function to train WiFi-based models
"""
#
##
import os
import time
import torch
import torch._dynamo
#
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score

from preset import preset

#

torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536

#
##
def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device,
          model_type: str,
          run_: int):
    
    """
    [description]
    : generic training function for WiFi-based models
    [parameter]
    : model: Pytorch model to train
    : optimizer: optimizer to train model (e.g., Adam)
    : loss: loss function to train model (e.g., BCEWithLogitsLoss)
    : data_train_set: training set
    : data_test_set: test set
    : var_threshold: threshold to binarize sigmoid outputs
    : var_batch_size: batch size of each training step
    : var_epochs: number of epochs to train model
    : device: device (cuda or cpu) to train model
    : model_type: type of model (e.g., lstm, cnn)
    : run_: run number of the model. Want to save every 2nd run.
    [return]
    : var_best_weight: weights of trained model
    """
    #
    ##
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle = True, pin_memory = True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))
    #
    ##
    var_best_accuracy = 0
    var_best_weight = None
    trials = 0  # counter for early stopping  
    #
    ##
    for var_epoch in range(var_epochs):
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_time_e0 = time.time()
        epoch_losses = []
        #
        model.train()
        #
        for data_batch in data_train_loader:
            #
            ##
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)
            #
            predict_train_y = model(data_batch_x)
            #
            var_loss_train = loss(predict_train_y, 
                                  data_batch_y.reshape(data_batch_y.shape[0], -1).float())
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            optimizer.step()
            #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ## Evaluate on training set
        var_accuracy_train, var_loss_train = evaluate(model, loss, data_train_loader, var_threshold, device)
        ## Evaluate on test set
        var_accuracy_test, var_loss_test = evaluate(model, loss, data_test_loader, var_threshold, device)
        #
        ## ---------------------------------------- Print -----------------------------------------
        #
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs"%(time.time() - var_time_e0),
              "- Loss %.6f"%var_loss_train.cpu(),
              "- Accuracy %.6f"%var_accuracy_train,
              "- Test Loss %.6f"%var_loss_test.cpu(),
              "- Test Accuracy %.6f"%var_accuracy_test)
        #
        ##
        if var_accuracy_test > var_best_accuracy:
            #
            trials = 0
            var_best_accuracy = var_accuracy_test
            var_best_weight = deepcopy(model.state_dict())
        else:
            #
            trials += 1
            if trials > preset["nn"]["early_stop"]:
                print("Early stopping at epoch", var_epoch)
                break
    #
    ##
    print("Best accuracy:", var_best_accuracy)
    print("Best loss:", var_loss_test)
    #
    if (run_+1) % 2 == 0:  # save the best weights after every 2 runs
        save_file_name = model_type + "_best_weight_run-" + str(run_+1) + ".pt"
        torch.save(var_best_weight, os.path.join(preset["path"]["model_wt"], save_file_name))

    return var_best_weight


def evaluate(model: Module,
             loss: Module,
             data_test_loader: DataLoader,
             var_threshold: float,
             device: device):
    """[description]
    : generic evaluation function for WiFi-based models
    [parameter]
    : model: Pytorch model to evaluate
    : loss: loss function to train model (e.g., BCEWithLogitsLoss)
    : data_test_loader: data loader for the set to be evaluated on
    : var_threshold: threshold to binarize sigmoid outputs
    : device: device (cuda or cpu) to train model
    [return]
    : var_accuracy_test: accuracy over the test set
    : var_loss_test: loss over the test set
    """
    
    model.eval()
    var_accuracy_test = []
    var_loss_test = []
    #
    with torch.no_grad():
        for test_data_batch in data_test_loader:
            data_test_x, data_test_y = test_data_batch
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            #
            predict_test_y = model(data_test_x)
            #
            var_loss_test.append(loss(predict_test_y, 
                                        data_test_y.reshape(data_test_y.shape[0], -1).float()))
            #
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
            #
            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()
            #
            predict_test_y = predict_test_y.reshape(data_test_y.shape[0], -1)
            data_test_y = data_test_y.reshape(data_test_y.shape[0], -1)
            #
            var_accuracy_test.append(accuracy_score(data_test_y.astype(int), 
                                                    predict_test_y.astype(int)))
    var_accuracy_test = sum(var_accuracy_test) / len(var_accuracy_test)
    var_loss_test = sum(var_loss_test) / len(var_loss_test)
    #
    return var_accuracy_test, var_loss_test
