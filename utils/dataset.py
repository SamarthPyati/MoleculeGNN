from typing import Optional, Any, List
from pathlib import Path
from enum import Enum

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

class RawDatasetList(str, Enum): 
    ESOL            = "ESOL.csv"
    LIPOPHILICITY   = "Lipophilicity.csv"
    FREESOLV        = "FreeSolv.csv"
    TOX21           = "Tox21.csv"

class MoleculeDataset(Dataset):
    def __init__(
        self, 
        csv_file: str, 
        smiles_col: str = 'smiles', 
        target_col: str = 'target',
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None
    ) -> None:
        """
        Initialize molecule dataset
        
        Args:
            csv_file: Path to CSV file containing SMILES and targets
            smiles_col: Name of column containing SMILES strings in csv
            target_col: Name of column containing target values in csv
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
        """
        super().__init__(None, transform, pre_transform)
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.file: str = csv_file
        self.smiles_col: str = smiles_col
        self.target_col: str = target_col
        
    def get(self, idx: int) -> Optional[Data]:
        """
        Get a single molecule graph
        
        Args:
            idx: Index of molecule to retrieve

        Returns:
            PyG Data object or None if invalid molecule
        """
        smiles: str = self.df.iloc[idx][self.smiles_col]
        target: float = self.df.iloc[idx][self.target_col]
        
        # Convert SMILES to graph
        mol: Optional[Mol] = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Extract features
        x: Tensor = self._get_node_features(mol)
        edge_index: Tensor = self._get_edge_index(mol)
        edge_attr: Tensor = self._get_edge_features(mol)
        y: Tensor = torch.tensor([target], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
    def _get_node_features(self, mol: Mol) -> Tensor:
        """
        Extract atom features from molecule
        """
        atom_features: List[List[float]] = []
        for atom in mol.GetAtoms():
            features: List[float] = self._get_atom_features(atom)
            atom_features.append(features)
        
        return torch.tensor(atom_features, dtype=torch.float)
    
    def _get_atom_features(self, atom: Atom) -> List[float]:
        """
        Extract features for a single atom
        """
        return [
            atom.GetAtomicNum(), 
            atom.GetDegree(), 
            atom.GetFormalCharge(), 
            atom.GetHybridization(), 
            atom.GetIsAromatic(), 
            atom.GetTotalNumHs(), 
        ]
    
    def _get_edge_index(self, mol: Mol) -> Tensor:
        """
        Extract bond connectivity from molecule
        """
        edge_indices: List[List[int]] = []
        
        for bond in mol.GetBonds():
            i: int = bond.GetBeginAtomIdx()
            j: int = bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if len(edge_indices) == 0:
            # Handle single-atom molecules
            return torch.zeros((2, 0), dtype=torch.long)
        
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    def _get_edge_features(self, mol: Mol) -> Tensor:
        """
        Extract bond features from molecule
        """
        edge_features: List[List[float]] = []
        
        for bond in mol.GetBonds():
            features: List[float] = self._get_bond_features(bond)
            # Add features for both directions
            edge_features.append(features)
            edge_features.append(features)
        
        if len(edge_features) == 0:
            # Handle single-atom molecules
            return torch.zeros((0, 3), dtype=torch.float)
        
        return torch.tensor(edge_features, dtype=torch.float)
    
    def _get_bond_features(self, bond: Bond) -> List[float]:
        """
        Extract features for a single bond
        """
        return [
            bond.GetBondTypeAsDouble(),
            bond.GetIsAromatic(),
            bond.IsInRing(),
        ]
    
    def len(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        filename: str = Path(self.file).name
        return f"MoleculeDataset(file={filename}, len={self.len()})"
