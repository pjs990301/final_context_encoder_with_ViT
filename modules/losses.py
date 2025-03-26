from torch.nn import functional as F
import torch
import torch.nn as nn

def get_loss(loss_name: str):
    
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()

    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()

    elif loss_name == 'mse':
        return nn.MSELoss()

    elif loss_name == 'l1':
        return nn.L1Loss()


    else:
        print(f'{loss_name}: invalid loss name')
        return