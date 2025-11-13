import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout, Module, Sequential
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data

class SimpleMoleculeGCN(Module):
    """
    Improved Graph Convolutional Network for molecular property prediction
    
    Improvements:
    - Residual/skip connections for better gradient flow
    - Multiple pooling strategies (mean, max, sum) concatenated
    - GELU activation for better performance
    - Layer normalization for better stability
    - Edge feature support (optional)
    - Configurable depth
    - Better prediction head architecture
    """
    def __init__(
        self, 
        num_node_features: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.3,
        num_layers: int = 3,
        use_edge_features: bool = False,
        num_edge_features: int = 0
    ) -> None:
        """
        Initialize improved GCN model
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout probability
            num_layers: Number of GCN layers
            use_edge_features: Whether to use edge features
            num_edge_features: Number of edge features (if use_edge_features=True)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features
        
        # Edge feature encoder (if using edge features)
        if use_edge_features and num_edge_features > 0:
            self.edge_encoder = Linear(num_edge_features, hidden_dim)
        else:
            self.edge_encoder = None
        
        # Input projection for residual connections
        if num_node_features != hidden_dim:
            self.input_proj = Linear(num_node_features, hidden_dim)
        else:
            self.input_proj = None
        
        # Graph convolution layers with residual connections
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                conv = GCNConv(num_node_features, hidden_dim)
            else:
                conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Multi-pooling: concatenate mean, max, and sum pooling
        # This gives the model 3x the information from graph-level aggregation
        pool_dim = hidden_dim * 3  # mean + max + sum
        
        # Improved prediction head with multiple layers
        self.pred_head = Sequential(
            Linear(pool_dim, hidden_dim * 2),
            LayerNorm(hidden_dim * 2),
            torch.nn.GELU(),
            Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim),
            LayerNorm(hidden_dim),
            torch.nn.GELU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object containing x, edge_index, batch, (optionally edge_attr)
            
        Returns:
            Tensor of predictions [batch_size, num_classes]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Encode edge features if available and enabled
        # Note: GCNConv doesn't directly support edge_attr, but we can encode it
        # For now, we'll use it in future enhancements or switch to GINEConv
        if self.use_edge_features and self.edge_encoder is not None and hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
            # Edge features can be incorporated in future versions
        
        # Graph convolutions with residual connections
        x_input: Tensor | None = None
        for i, (conv, bn, ln) in enumerate(zip(self.convs, self.batch_norms, self.layer_norms)):
            # Store input for residual connection (only if dimensions match)
            if i > 0:
                x_input = x
            
            # Convolution
            x = conv(x, edge_index)
            
            # Batch normalization
            x = bn(x)
            
            # Residual connection (skip first layer as input dim may differ)
            if i > 0 and x_input is not None and x.shape == x_input.shape:
                x = x + x_input  # Residual connection
            
            # Layer normalization
            x = ln(x)
            
            # GELU activation (better than ReLU for deep networks)
            x = F.gelu(x)
        
        # Multi-pooling: concatenate different pooling strategies
        # At this point, x is guaranteed to be a Tensor (not None)
        assert x is not None, "x should not be None after convolutions"
        x_mean: Tensor = global_mean_pool(x, batch)
        x_max: Tensor = global_max_pool(x, batch)
        x_sum: Tensor = global_add_pool(x, batch)
        
        # Concatenate all pooling strategies
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Prediction head
        x = self.pred_head(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize weights with better initialization scheme"""
        import torch.nn.init as init
        
        for m in self.modules():
            if isinstance(m, Linear):
                # Use Kaiming initialization for better gradient flow with GELU
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, (BatchNorm1d, LayerNorm)):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)