from typing import List, Optional
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Mol
from core.dataset import get_atom_features


class ModelPredictor:
    """Class for making predictions on new molecules"""
    
    def __init__(
        self,
        model: Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initialize predictor
        
        Args:
            model: Trained PyTorch model
            device: Device to use
        """
        self.model: Module = model.to(device)
        self.device: str = device
        self.model.eval()
        
    def predict_smiles(
        self,
        smiles: str,
        return_probability: bool = True
    ) -> Optional[float]:
        """
        Predict property for a single SMILES string
        
        Args:
            smiles: SMILES string
            return_probability: Apply sigmoid for classification
            
        Returns:
            Prediction value or None if invalid SMILES
        """
        mol: Optional[Mol] = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Convert to graph
        data: Data = self._smiles_to_graph(smiles)
        data = data.to(self.device)

        with torch.no_grad():
            out: Tensor = self.model(data)
            
        if return_probability:
            out = torch.sigmoid(out)
            
        return out.item()
    
    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        return_probability: bool = True
    ) -> np.ndarray:
        """
        Predict properties for multiple SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            return_probability: Apply sigmoid for classification
            
        Returns:
            Array of predictions
        """
        # Convert SMILES to graphs
        dataset: List[Data] = []
        valid_indices: List[int] = []
        
        for idx, smiles in enumerate(smiles_list):
            data = self._smiles_to_graph(smiles)
            if data is not None:
                dataset.append(data)
                valid_indices.append(idx)
        
        # Create data loader
        loader: DataLoader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        # Make predictions
        predictions: List[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out: Tensor = self.model(batch)
                if return_probability:
                    out = torch.sigmoid(out)
                predictions.append(out.cpu())
        
        predictions_tensor: Tensor = torch.cat(predictions, dim=0)
        
        # Fill results including invalid molecules
        results: np.ndarray = np.full(len(smiles_list), np.nan)
        results[valid_indices] = predictions_tensor.numpy().flatten()
        
        return results
    
    def _smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES to graph
        
        Args:
            smiles: SMILES string
            
        Returns:
            PyG Data object or None
        """
        mol: Optional[Mol] = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Extract features of atoms in molecule (using MoleculeDataset methods)
        atom_features: List[List[float]] = [get_atom_features(atom) for atom in mol.GetAtoms()]
        
        x: Tensor = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract edges
        edge_index: List[List[int]] = []
        for bond in mol.GetBonds():
            i: int = bond.GetBeginAtomIdx()
            j: int = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        if len(edge_index) == 0:
            edge_index_tensor: Tensor = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index_tensor: Tensor = torch.tensor(
                edge_index, dtype=torch.long
            ).t().contiguous()
        
        return Data(x=x, edge_index=edge_index_tensor)
