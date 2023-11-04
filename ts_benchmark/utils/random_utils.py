import random

import numpy as np
import torch
SEED = 2021

def fix_random_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)