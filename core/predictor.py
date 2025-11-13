from typing import List, Optional
from config.config import get_device_for_torch
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Mol
from core.dataset import get_atom_features, get_bond_features


class ModelPredictor:
    """Class for making predictions on new molecules"""
    
    def __init__(
        self,
        model: Module,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize predictor
        
        Args:
            model: Trained PyTorch model
            device: Device to use (auto-detected if None)
        """
        self.device: str = device or get_device_for_torch()
        self.model: Module = model.to(self.device)
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
        data: Optional[Data] = self._smiles_to_graph(smiles)
        if data is None:
            return None
        
        # Add batch attribute for single molecule (required for pooling)
        if data.x is not None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        else:
            return None
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
        
        # Extract edges and edge features
        edge_index: List[List[int]] = []
        edge_features: List[List[float]] = []
        
        for bond in mol.GetBonds():
            i: int = bond.GetBeginAtomIdx()
            j: int = bond.GetEndAtomIdx()
            bond_features: List[float] = get_bond_features(bond)
            
            # Add both directions (undirected graph)
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_features.append(bond_features)
            edge_features.append(bond_features)
        
        if len(edge_index) == 0:
            edge_index_tensor: Tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor: Tensor = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(
                edge_index, dtype=torch.long
            ).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
