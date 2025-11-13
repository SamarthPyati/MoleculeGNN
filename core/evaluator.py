from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Mol

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(
        self,
        model: Module,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize evaluator
        
        Args:
            model: Trained PyTorch model
            device: Device to use (auto-detected if None)
        """
        from config.config import get_device_for_torch
        self.device: str = device or get_device_for_torch()
        self.model: Module = model.to(self.device)
        self.model.eval()
        
    def predict(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset
        
        Args:
            loader: Data loader
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        predictions: List[Tensor] = []
        targets: List[Tensor] = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out: Tensor = self.model(batch)
                predictions.append(out.cpu())
                targets.append(batch.y.cpu())
        
        predictions_tensor: Tensor = torch.cat(predictions, dim=0)
        targets_tensor: Tensor = torch.cat(targets, dim=0)
        
        return predictions_tensor.numpy(), targets_tensor.numpy()
    
    def evaluate_classification(
        self,
        loader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate classification performance
        
        Args:
            loader: Data loader
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        predictions, targets = self.predict(loader)
        
        # Apply sigmoid and threshold
        predictions_prob: np.ndarray = 1 / (1 + np.exp(-predictions))
        predictions_binary: np.ndarray = (predictions_prob > threshold).astype(int)
        targets_int: np.ndarray = targets.astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets_int, predictions_binary),
            'precision': precision_score(targets_int, predictions_binary, zero_division='warn'),
            'recall': recall_score(targets_int, predictions_binary, zero_division='warn'),
            'f1': f1_score(targets_int, predictions_binary, zero_division='warn'),
            'roc_auc': roc_auc_score(targets, predictions_prob),
        }
        
        # Confusion matrix
        cm: np.ndarray = confusion_matrix(targets_int, predictions_binary)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def evaluate_regression(
        self,
        loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate regression performance
        
        Args:
            loader: Data loader
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        predictions, targets = self.predict(loader)
        
        from sklearn.metrics import root_mean_squared_error
        metrics: Dict[str, float] = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': root_mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            loader: Data loader
            threshold: Classification threshold
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        predictions, targets = self.predict(loader)
        
        # Apply sigmoid and threshold
        predictions_prob: np.ndarray = 1 / (1 + np.exp(-predictions))
        predictions_binary: np.ndarray = (predictions_prob > threshold).astype(int)
        targets_int: np.ndarray = targets.astype(int).flatten()
        
        cm: np.ndarray = confusion_matrix(targets_int, predictions_binary.flatten())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(
        self,
        loader: DataLoader,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve
        
        Args:
            loader: Data loader
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        
        predictions, targets = self.predict(loader)
        predictions_prob: np.ndarray = 1 / (1 + np.exp(-predictions))
        
        fpr: np.ndarray
        tpr: np.ndarray
        thresholds: np.ndarray
        fpr, tpr, thresholds = roc_curve(targets, predictions_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(
        self,
        loader: DataLoader,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of predictions
        
        Args:
            loader: Data loader
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        predictions, targets = self.predict(loader)
        predictions_prob: np.ndarray = 1 / (1 + np.exp(-predictions))
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            predictions_prob[targets == 0],
            bins=50, alpha=0.5, label='Negative Class', color='blue'
        )
        plt.hist(
            predictions_prob[targets == 1],
            bins=50, alpha=0.5, label='Positive Class', color='red'
        )
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Prediction Distribution by Class')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def visualize_molecule(
    smiles: str,
    save_path: Optional[str] = None,
    size: Tuple[int, int] = (400, 400)
) -> Any:
    """
    Visualize a molecule from SMILES
    
    Args:
        smiles: SMILES string
        save_path: Path to save image
        size: Image size (width, height)
        
    Returns:
        PIL Image object
    """
    from rdkit.Chem import Draw
    
    mol: Optional[Mol] = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    
    img = Draw.MolToImage(mol, size=size)
    
    if save_path:
        img.save(save_path)
    
    return img


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_metric'
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Metric plot
    axes[1].plot(history['val_metric'], label='Validation Metric', 
                linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].set_title('Validation Performance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()