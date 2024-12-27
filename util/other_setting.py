import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("CUDA cache cleared.")


def get_backbone_gradients(model):
    backbone_gradients = []
    for name, param in model.backbone.named_parameters():
        if param.grad is not None:
            backbone_gradients.append(param.grad.detach().clone())
    return backbone_gradients
