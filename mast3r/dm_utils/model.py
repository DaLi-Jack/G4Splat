import torch
import numpy as np

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
