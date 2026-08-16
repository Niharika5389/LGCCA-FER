"""Deterministic seeding utility.

Shared by any training/preprocessing script in this project that needs
reproducible results (random, numpy, torch CPU/CUDA).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility.

    Also configures cuDNN to run in deterministic mode. Note that fully
    deterministic behavior may come at some performance cost.

    Args:
        seed: The random seed to use everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
