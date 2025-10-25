from typing import Tuple

import numpy as np
import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from core import MoleculeDataset

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
    
    # Create loaders
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
