import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    # Set all rng seed to be deterministic
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cuda compatible machines
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False