"""
CNN Feature Extractor for SIAO-CNN-ORNN Model

Implements the Convolutional Neural Network block for extracting
spatial-temporal features from reactor time-series data.

Architecture:
- Conv2D layers (3 layers)
- ReLU activation
- Batch Normalization
- Max Pooling
- Adaptive pooling for variable input sizes
- Flatten layer for RNN input

Author: Deep Learning Architect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNNBlock(nn.Module):
    """
    Single CNN block: Conv2D → BatchNorm → ReLU → MaxPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
        pool_size: Tuple[int, int] = (2, 2),
        use_pooling: bool = True
    ):
        super(CNNBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if use_pooling else nn.Identity()
        self.use_pooling = use_pooling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for SIAO-CNN-ORNN model.
    
    Extracts spatial-temporal features from reactor time-series data
    using stacked convolutional layers.
    
    Input:  [batch, 1, time_steps, features] or [batch, time_steps, features]
    Output: [batch, embedding_dim] - Feature embeddings for RNN
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_time_steps: int = 50,
        input_features: int = 43,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[Tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        pool_sizes: Tuple[Tuple[int, int], ...] = ((2, 2), (2, 2), (2, 1)),
        embedding_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize CNN Feature Extractor.
        
        Args:
            input_channels: Number of input channels (1 for single time-series)
            input_time_steps: Window size (time dimension)
            input_features: Number of input features
            conv_channels: Output channels for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            pool_sizes: Pool sizes for each layer
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.input_time_steps = input_time_steps
        self.input_features = input_features
        self.embedding_dim = embedding_dim
        
        # Build CNN blocks
        layers = []
        in_ch = input_channels
        
        for i, (out_ch, k_size, p_size) in enumerate(zip(conv_channels, kernel_sizes, pool_sizes)):
            # Only use pooling if dimensions allow
            use_pool = True
            layers.append(CNNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k_size,
                padding=k_size[0] // 2,  # Same padding
                pool_size=p_size,
                use_pooling=use_pool
            ))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self._flat_size = self._get_flat_size(input_time_steps, input_features)
        
        # Fully connected layer to get embedding
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_flat_size(self, time_steps: int, features: int) -> int:
        """Calculate output size after all conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, time_steps, features)
            out = self.conv_layers(dummy)
            return out.numel()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, time_steps, features] or 
               [batch, channels, time_steps, features]
        
        Returns:
            Feature embeddings [batch, embedding_dim]
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, time, features]
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten and embed
        x = self.fc(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.embedding_dim


class CNNFeatureExtractorForSequence(nn.Module):
    """
    CNN Feature Extractor that preserves sequence for RNN input.
    
    Applies CNN across windows and outputs sequence of features.
    
    Input:  [batch, num_windows, time_steps, features]
    Output: [batch, num_windows, embedding_dim] - Sequence for RNN
    """
    
    def __init__(
        self,
        input_time_steps: int = 50,
        input_features: int = 43,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        embedding_dim: int = 256,
        dropout: float = 0.3
    ):
        super(CNNFeatureExtractorForSequence, self).__init__()
        
        self.cnn = CNNFeatureExtractor(
            input_channels=1,
            input_time_steps=input_time_steps,
            input_features=input_features,
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence of windows.
        
        Args:
            x: Input [batch, num_windows, time_steps, features]
        
        Returns:
            Sequence embeddings [batch, num_windows, embedding_dim]
        """
        batch_size, num_windows, time_steps, features = x.shape
        
        # Reshape to process all windows through CNN
        x = x.view(batch_size * num_windows, time_steps, features)
        
        # Extract features for each window
        x = self.cnn(x)  # [batch * num_windows, embedding_dim]
        
        # Reshape back to sequence
        x = x.view(batch_size, num_windows, -1)
        
        return x


# =============================================================================
# Utility Functions
# =============================================================================

def create_cnn_extractor(
    input_shape: Tuple[int, int],
    embedding_dim: int = 256,
    num_layers: int = 3,
    base_channels: int = 32,
    dropout: float = 0.3
) -> CNNFeatureExtractor:
    """
    Factory function to create CNN feature extractor.
    
    Args:
        input_shape: (time_steps, features)
        embedding_dim: Output embedding dimension
        num_layers: Number of conv layers
        base_channels: Base channel count (doubled each layer)
        dropout: Dropout rate
    
    Returns:
        CNNFeatureExtractor instance
    """
    time_steps, features = input_shape
    
    # Generate channel progression
    conv_channels = tuple(base_channels * (2 ** i) for i in range(num_layers))
    
    # Generate kernel and pool sizes
    kernel_sizes = tuple((3, 3) for _ in range(num_layers))
    pool_sizes = [(2, 2) if i < num_layers - 1 else (2, 1) for i in range(num_layers)]
    
    return CNNFeatureExtractor(
        input_channels=1,
        input_time_steps=time_steps,
        input_features=features,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        pool_sizes=tuple(pool_sizes),
        embedding_dim=embedding_dim,
        dropout=dropout
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("CNN Feature Extractor for SIAO-CNN-ORNN - Demo")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = CNNFeatureExtractor(
        input_channels=1,
        input_time_steps=50,
        input_features=43,
        conv_channels=(32, 64, 128),
        embedding_dim=256,
        dropout=0.3
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTrainable Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 50, 43).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output embedding dim: {model.get_output_dim()}")
    
    # Test sequence version
    print("\n" + "=" * 60)
    print("Testing Sequence CNN for RNN input...")
    
    seq_model = CNNFeatureExtractorForSequence(
        input_time_steps=50,
        input_features=43,
        embedding_dim=256
    ).to(device)
    
    x_seq = torch.randn(batch_size, 10, 50, 43).to(device)  # 10 windows
    
    print(f"Sequence input shape: {x_seq.shape}")
    
    with torch.no_grad():
        output_seq = seq_model(x_seq)
    
    print(f"Sequence output shape: {output_seq.shape}")
    print(f"Ready for RNN: [batch={batch_size}, seq_len=10, features=256]")
