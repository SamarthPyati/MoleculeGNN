import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, BatchNorm1d, Dropout, Module
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class SimpleMoleculeGCN(Module):
    """
    Simple Graph Convolutional Network for molecular property prediction
    """
    def __init__(
        self, 
        num_node_features: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.3
    ) -> None:
        """
        Initialize GCN model
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Graph convolution layers
        self.conv1: GCNConv = GCNConv(num_node_features, hidden_dim)
        self.bn1: BatchNorm1d = BatchNorm1d(hidden_dim)
        
        self.conv2: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.bn2: BatchNorm1d = BatchNorm1d(hidden_dim)
        
        self.conv3: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.bn3: BatchNorm1d = BatchNorm1d(hidden_dim)
        
        # Prediction layers
        self.fc1: Linear = Linear(hidden_dim, hidden_dim // 2)
        self.dropout: Dropout = Dropout(dropout)
        self.fc2: Linear = Linear(hidden_dim // 2, num_classes)
        
        # Initialize weights to prevent extreme values
        self._initialize_weights()
        
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object containing x, edge_index, batch
            
        Returns:
            Tensor of predictions [batch_size, num_classes]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Prediction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier uniform for better stability"""
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)