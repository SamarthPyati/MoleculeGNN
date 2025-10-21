from typing import List, Tuple, Dict, Optional

from config import TrainingConfig

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from tqdm import tqdm

class ModelTrainer:
    """Trainer class for molecular property prediction"""
    def __init__(
        self,
        model: Module,
        config: TrainingConfig = TrainingConfig(),
        device: Optional[str] = None
    ) -> None:
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config: TrainingConfig = config
        self.device: str = device or config.device
        self.model: Module = model.to(self.device)
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': []
        }
        
    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: Optimizer,
        criterion: Module
    ) -> float:
        """
        Train for one epoch
        
        Args:
            loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss: float = 0.0
        
        for batch in tqdm(loader, desc='Training'):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            out: Tensor = self.model(batch)
            # Ensure target shape matches model output (N, 1) for BCEWithLogitsLoss
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss: Tensor = criterion(out, targets.float())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
        return total_loss / len(loader.dataset)
    
    def evaluate(
        self,
        loader: DataLoader,
        criterion: Module,
        task: str = 'classification'
    ) -> Tuple[float, float, str]:
        """
        Evaluate model on validation/test set
        
        Args:
            loader: Data loader
            criterion: Loss function
            task: 'classification' or 'regression'
            
        Returns:
            Tuple of (loss, metric, metric_name)
        """
        self.model.eval()
        total_loss: float = 0.0
        predictions: List[Tensor] = []
        targets: List[Tensor] = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out: Tensor = self.model(batch)
                # match shapes as in training
                batch_targets = batch.y
                if batch_targets.dim() == 1:
                    batch_targets = batch_targets.unsqueeze(1)
                loss: Tensor = criterion(out, batch_targets.float())
                
                total_loss += loss.item() * batch.num_graphs
                predictions.append(out.cpu())
                targets.append(batch_targets.cpu())
        
        predictions_tensor: Tensor = torch.cat(predictions, dim=0)
        targets_tensor: Tensor = torch.cat(targets, dim=0)
        
        avg_loss: float = total_loss / len(loader.dataset)
        
        # Calculate metric based on task
        if task == 'classification':
            # flatten to shape (N,) for roc_auc_score when single-output
            predictions_prob: np.ndarray = torch.sigmoid(predictions_tensor).numpy().ravel()
            targets_np: np.ndarray = targets_tensor.numpy().ravel()
            # Ensure binary labels for roc_auc_score. If labels are continuous/probabilistic,
            # threshold at 0.5. If there are >2 distinct values after thresholding, raise.
            # This avoids "continuous format is not supported".
            if not np.array_equal(np.unique(targets_np), np.unique(targets_np).astype(int)):
                # threshold probabilistic labels to binary
                targets_bin = (targets_np >= 0.5).astype(int)
            else:
                targets_bin = targets_np.astype(int)

            # final sanity check: roc_auc requires exactly two label classes
            uniq = np.unique(targets_bin)
            if uniq.size != 2:
                raise ValueError(
                    f'ROC-AUC requires binary labels but got {uniq}. '
                    'If this is a regression task set task="regression".'
                )

            metric: float = roc_auc_score(targets_bin, predictions_prob)
            metric_name: str = 'ROC-AUC'
        else:  # regression
            predictions_np: np.ndarray = predictions_tensor.numpy().ravel()
            targets_np: np.ndarray = targets_tensor.numpy().ravel()
            metric: float = root_mean_squared_error(
                targets_np, predictions_np
            )
            metric_name: str = 'RMSE'
            
        return avg_loss, metric, metric_name
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        task: Optional[str] = None,
        patience: Optional[int] = None
    ) -> None:
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config if None)
            lr: Learning rate (uses config if None)
            task: Task type (uses config if None)
            patience: Early stopping patience (uses config if None)
        """
        epochs = epochs or self.config.epochs
        lr = lr or self.config.learning_rate
        task = task or self.config.task
        patience = patience or self.config.patience
        
        optimizer: Optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        if task == 'classification':
            criterion: Module = torch.nn.BCEWithLogitsLoss()
        else:
            criterion: Module = torch.nn.MSELoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss: float = float('inf')
        patience_counter: int = 0
        
        for epoch in range(epochs):
            train_loss: float = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_metric, metric_name = self.evaluate(
                val_loader, criterion, task
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(val_metric)
            
            scheduler.step(val_loss)
            
            print(
                f'Epoch {epoch+1:03d}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val {metric_name}: {val_metric:.4f}'
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        print(f'Training completed. Best validation loss: {best_val_loss:.4f}')