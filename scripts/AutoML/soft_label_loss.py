import torch
from torch import nn
import torch.nn.functional as F

def soft_label_loss(output,label):
    K = label.shape[-1]

