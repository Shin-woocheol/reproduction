import numpy as np
import torch
import random

def fix_seed(seed = 777):
    random.seed(seed) # python random
    np.random.seed(seed) # numpy random
    torch.manual_seed(seed) # pytorch random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #single GPU
        torch.cuda.manual_seed_all(seed) #multi GPU
        torch.backends.cudnn.deterministic = True #cudnn non deterministic off
        torch.backends.cudnn.benchmark = False #cudnn 성능 최적화 off