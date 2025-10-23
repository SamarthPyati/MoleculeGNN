import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, BatchNorm1d, Dropout, Module
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data

class AdvancedMoleculeGNN(Module):
    """
    Advanced GNN with edge features using GIN architecture
    
    GIN (Graph Isomorphism Network) is more expressive than GCN
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.3
    ) -> None:
        """
        Initialize advanced GNN model
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # GIN layers
        nn1 = torch.nn.Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv1: GINConv = GINConv(nn1)
        
        nn2 = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv2: GINConv = GINConv(nn2)
        
        nn3 = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv3: GINConv = GINConv(nn3)
        
        # Edge network
        self.edge_encoder: Linear = Linear(num_edge_features, hidden_dim)
        
        # Prediction head
        self.fc1: Linear = Linear(hidden_dim, hidden_dim // 2)
        self.dropout: Dropout = Dropout(dropout)
        self.fc2: Linear = Linear(hidden_dim // 2, num_classes)
        
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object
            
        Returns:
            Tensor of predictions
        """
        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        edge_attr: Tensor = data.edge_attr
        batch: Tensor = data.batch
        
        # Encode edges
        edge_embedding: Tensor = self.edge_encoder(edge_attr)
        
        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Aggregate
        x = global_add_pool(x, batch)
        
        # Predict
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x