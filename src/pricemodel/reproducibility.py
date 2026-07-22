"""Random-state controls shared by training and retraining entry points."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, and return a seeded DataLoader generator.

    Runs are reproducible on the same hardware/software stack. Exact equality
    across CPU, CUDA, and MPS is not guaranteed because their kernels differ.
    """
    if seed < 0:
        raise ValueError("random_seed must be zero or positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
