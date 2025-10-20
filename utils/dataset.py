from typing import Optional
from pathlib import Path

import pandas as pd
from enum import Enum
from rdkit import Chem

import torch
from torch_geometric.data import Data, Dataset

class RawDatasetList(str, Enum): 
    ESOL            = "ESOL.csv"
    LIPOPHILICITY   = "Lipophilicity.csv"
    FREESOLV        = "FreeSolv.csv"
    TOX21           = "Tox21.csv"

class MoleculeDataset(Dataset):
    def __init__(self, csv_file, smiles_col='smiles', target_col='target'):
        """
        csv_file: Path to CSV file for raw data
        smiles_col: Column name for 'smiles' strings
        target_col: Column name for target values (solubility, toxicity)
        """
        super().__init__()
        self.file = csv_file
        self.df = pd.read_csv(csv_file)
        self.smiles_col = smiles_col
        self.target_col = target_col

    @property
    def len(self):
        return len(self.df)
    
    def get(self, idx: int) -> Optional[Data]:
        smiles = self.df.iloc[idx][self.smiles_col]
        target = self.df.iloc[idx][self.target_col]
        
        # Convert SMILES to graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self.get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract bonds (edges)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            bond_features = self.get_bond_features(bond)
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Target
        y = torch.tensor([target], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def get_atom_features(self, atom):
        """Extract features for a single atom"""
        return [
            atom.GetAtomicNum(),                # Atomic number
            atom.GetDegree(),                   # Number of bonds
            atom.GetFormalCharge(),             # Charge
            atom.GetHybridization().real,       # Hybridization
            atom.GetIsAromatic(),               # Aromaticity
            atom.GetTotalNumHs(),               # Number of hydrogens
        ]
    
    def get_bond_features(self, bond):
        """Extract features for a single bond"""
        return [
            bond.GetBondTypeAsDouble(),     # Bond type (1, 2, 3, 1.5)
            bond.GetIsAromatic(),           
            bond.IsInRing(),                
        ]
    
    def __repr__(self) -> str:
        filename = Path(self.file).name
        return f"MoleculeDataset(file={str(filename)}, len={self.len})"