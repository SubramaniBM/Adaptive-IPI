"""
src/utils/reproducibility.py

Reproducibility utilities for the Adaptive-IPI project.

Ensures deterministic behaviour across runs by seeding all
relevant random number generators.
"""

import os
import random

import numpy as np
import torch

from src.core.constants import SEED


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Also configures PyTorch for deterministic operations where possible.

    Args:
        seed: Random seed value. Defaults to the project-wide SEED constant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8 deterministic algorithms flag
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns:
        A torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
