from typing import Tuple, Dict, Optional, Any

import numpy as np
import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from core import MoleculeDataset
from config import ModelConfig, get_device_for_torch

from models import SimpleMoleculeGCN

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
        # Enable benchmark for faster training (disable if reproducibility is critical)
        torch.backends.cudnn.benchmark = True

def count_parameters(model: Module) -> int:
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(
    csv_file: str,
    smiles_col: str = 'smiles',
    target_col: str = 'target'
) -> MoleculeDataset:
    """
    Load molecule dataset from CSV
    
    Args:
        csv_file: Path to CSV file
        smiles_col: SMILES column name
        target_col: Target column name
        
    Returns:
        MoleculeDataset instance
    """
    return MoleculeDataset(csv_file, smiles_col, target_col)

def create_data_loaders(
    dataset: MoleculeDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders
    
    Args:
        dataset: MoleculeDataset instance
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size
        num_workers: Number of workers for data loading
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Remove invalid molecules
    valid_dataset = [d for d in dataset if d is not None]
    
    # Split Dataset
    test_ratio: float = (1 - train_ratio)
    train_dataset, temp = train_test_split(
        valid_dataset,
        test_size=test_ratio,
        random_state=seed
    )
    
    val_size: float = val_ratio / train_ratio
    val_dataset, test_dataset = train_test_split(
        temp,
        test_size=(1 - val_size),
        random_state=seed
    )
    
    # Determine if we should use pin_memory (only for CUDA, not MPS)
    import torch
    use_pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    # For MPS, use fewer workers to avoid overhead (MPS handles parallelism differently)
    # For CUDA, use more workers for better data loading
    effective_num_workers = num_workers if torch.cuda.is_available() else min(num_workers, 2)
    
    # Create loaders with optimizations for GPU training
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
        drop_last=False
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
        drop_last=False
    )
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def save_model(
    model: Module,
    path: str,
    config: Optional[ModelConfig] = None
) -> None:
    """
    Save model and configuration into a checkpoint
    
    Args:
        model: PyTorch model
        path: Save path
        config: Model configuration
    """
    checkpoint: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, path)
    print(f'Model saved to {path}')


def load_model(
    path: str,
    model_class: type = SimpleMoleculeGCN,
    device: Optional[str] = None
) -> Module:
    """
    Load model from checkpoint
    
    Args:
        path: Path to checkpoint
        model_class: Model class to instantiate
        device: Device to load model to (auto-detected if None)
        
    Returns:
        Loaded model
    """
    device = device or get_device_for_torch()
    checkpoint: Dict[str, Any] = torch.load(path, map_location=torch.device(device))
    
    # Handle both checkpoint format and direct state_dict
    model: Module
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_config = checkpoint.get('config', None)
        if saved_config is not None:
            config: ModelConfig = saved_config
            model = model_class(**vars(config))
        else:
            # Use default configuration if not saved
            model = model_class(num_node_features=6, hidden_dim=64, num_classes=1)
        model.load_state_dict(state_dict)
    else:
        # Direct state_dict format - need to infer model architecture
        # This is a fallback - ideally models should be saved with config
        model = model_class(num_node_features=6, hidden_dim=64, num_classes=1)
        model.load_state_dict(checkpoint)
    
    model = model.to(torch.device(device))
    model.eval()
    
    print(f'Model loaded from {path}')
    return model