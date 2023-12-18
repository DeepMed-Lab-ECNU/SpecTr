import csv
import numpy as np
import torch
import shutil
import torch.nn as nn

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def save_dict(save_file:str, dict_obj: dict):
    with open(save_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for key in dict_obj:
            writer.writerow([key, dict_obj[key]])

def save_pyfile(save_file:str, out_file:str):
    shutil.copy(save_file, out_file)


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

    
def convert_2d_gn(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Get current bn layer
            bn = get_layer(model, name)
            # Create new gn layer
            if bn.num_features == 1:
                gn = nn.GroupNorm(1, bn.num_features)
            else:
                gn = nn.GroupNorm(8, bn.num_features)
            # Assign gn
            print("Swapping {} with {}".format(bn, gn))
            set_layer(model, name, gn)
    return model
    

