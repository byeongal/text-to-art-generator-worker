import gc
import random

import loguru
import numpy as np
import torch
from settings import torch_model_settings


def clear_memory() -> None:
    """
    Function to clear memory and gpu memory
    """
    loguru.logger.info("`clear_memory` was called")
    gc.collect()
    if torch_model_settings.device == torch.device("cuda"):
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """
    Set random seed.
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
