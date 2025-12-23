import random
import numpy as np
import torch
import os


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value to use for all random number generators
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch deterministic operations (may reduce performance)
    # Uncomment if you need full reproducibility (slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For transformers library reproducibility
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

