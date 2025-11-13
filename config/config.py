from dataclasses import dataclass

def get_device_for_torch() -> str:
    # Set device accelerator for the current device
    import torch 
    if torch.mps.is_available():    # MacOS
        return 'mps'
    elif torch.cuda.is_available(): # CUDA 
        return 'cuda'
    else: 
        return 'cpu'

@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 15
    device: str = get_device_for_torch()
    num_workers: int = 4
    task: str = 'classification'  # 'classification' or 'regression'

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    num_node_features: int = 6
    num_edge_features: int = 3
    hidden_dim: int = 128
    num_classes: int = 1
    num_layers: int = 3
    dropout: float = 0.3
    pooling: str = 'mean'  # 'mean', 'add', 'max'