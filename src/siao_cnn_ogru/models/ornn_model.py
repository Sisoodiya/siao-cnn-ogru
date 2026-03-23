"""
Optimized RNN (ORNN) with SIAO Weight Optimization

Implements a Recurrent Neural Network whose initial weights are
optimized using Self-Improved Aquila Optimizer (SIAO) followed
by fine-tuning with backpropagation.

Architecture:
- GRU-based recurrent layer (configurable to vanilla RNN)
- SIAO optimization for weight initialization
- Backpropagation fine-tuning
- Output embeddings for classification head

Author: Recurrent Neural Network Specialist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm, trange
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Use Console instead of standard logger for visual consistency
console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Optimized RNN Cell
# =============================================================================

class ORNNCell(nn.Module):
    """
    Optimized RNN Cell with exposed weights for SIAO optimization.
    
    Implements: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ):
        super(ORNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable parameters (exposed for SIAO)
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        if self.b_ih is not None:
            nn.init.zeros_(self.b_ih)
            nn.init.zeros_(self.b_hh)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, input_size]
            h: Hidden state [batch, hidden_size]
        
        Returns:
            New hidden state [batch, hidden_size]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # RNN computation
        h_new = torch.tanh(
            F.linear(x, self.W_ih, self.b_ih) +
            F.linear(h, self.W_hh, self.b_hh)
        )
        
        return h_new
    
    def get_weight_vector(self) -> np.ndarray:
        """Flatten all weights into a single vector for SIAO."""
        weights = []
        weights.append(self.W_ih.data.cpu().numpy().flatten())
        weights.append(self.W_hh.data.cpu().numpy().flatten())
        if self.b_ih is not None:
            weights.append(self.b_ih.data.cpu().numpy())
            weights.append(self.b_hh.data.cpu().numpy())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from flattened vector."""
        idx = 0
        
        # W_ih
        size = self.hidden_size * self.input_size
        self.W_ih.data = torch.tensor(
            weights[idx:idx+size].reshape(self.hidden_size, self.input_size),
            dtype=torch.float32, device=device
        )
        idx += size
        
        # W_hh
        size = self.hidden_size * self.hidden_size
        self.W_hh.data = torch.tensor(
            weights[idx:idx+size].reshape(self.hidden_size, self.hidden_size),
            dtype=torch.float32, device=device
        )
        idx += size
        
        # Biases
        if self.b_ih is not None:
            size = self.hidden_size
            self.b_ih.data = torch.tensor(
                weights[idx:idx+size], dtype=torch.float32, device=device
            )
            idx += size
            self.b_hh.data = torch.tensor(
                weights[idx:idx+size], dtype=torch.float32, device=device
            )


# =============================================================================
# Optimized GRU Cell
# =============================================================================

class OGRUCell(nn.Module):
    """
    Optimized GRU Cell with exposed weights for SIAO optimization.
    
    GRU equations:
    r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1} + b_r)
    z_t = sigmoid(W_iz * x_t + W_hz * h_{t-1} + b_z)
    n_t = tanh(W_in * x_t + r_t * (W_hn * h_{t-1}) + b_n)
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ):
        super(OGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input weights (3 gates: reset, update, new)
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_iz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_in = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # Hidden weights
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
            self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
            self.b_in = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hz = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hn = nn.Parameter(torch.Tensor(hidden_size))
        else:
            for name in ['b_ir', 'b_iz', 'b_in', 'b_hr', 'b_hz', 'b_hn']:
                self.register_parameter(name, None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'W_' in name:
                nn.init.xavier_uniform_(param)
            elif 'b_' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # Reset gate
        r = torch.sigmoid(
            F.linear(x, self.W_ir, self.b_ir) +
            F.linear(h, self.W_hr, self.b_hr)
        )
        
        # Update gate
        z = torch.sigmoid(
            F.linear(x, self.W_iz, self.b_iz) +
            F.linear(h, self.W_hz, self.b_hz)
        )
        
        # New gate
        n = torch.tanh(
            F.linear(x, self.W_in, self.b_in) +
            r * F.linear(h, self.W_hn, self.b_hn)
        )
        
        # Hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new
    
    def get_weight_vector(self) -> np.ndarray:
        """Flatten all weights for SIAO."""
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from vector."""
        idx = 0
        for name, param in self.named_parameters():
            size = param.numel()
            param.data = torch.tensor(
                weights[idx:idx+size].reshape(param.shape),
                dtype=torch.float32, device=device
            )
            idx += size


# =============================================================================
# Self-Attention Layer
# =============================================================================

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for time-series.
    
    Computes attention weights across the sequence dimension
    to focus on important timesteps.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            
        Returns:
            Attended output [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        
        return self.out(out)


# =============================================================================
# Optimized RNN (ORNN) Module
# =============================================================================

class ORNN(nn.Module):
    """
    Optimized RNN with SIAO weight initialization.
    
    Supports both vanilla RNN and GRU variants.
    Output can be used as input to fully connected classifier.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = 'gru',  # 'rnn' or 'gru'
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(ORNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Build RNN layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            if cell_type == 'gru':
                cell = OGRUCell(layer_input_size, hidden_size)
            else:
                cell = ORNNCell(layer_input_size, hidden_size)
            
            self.cells.append(cell)
            
            if bidirectional:
                if cell_type == 'gru':
                    cell_bw = OGRUCell(layer_input_size, hidden_size)
                else:
                    cell_bw = ORNNCell(layer_input_size, hidden_size)
                self.cells.append(cell_bw)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        logger.info(f"ORNN: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, type={cell_type}, bidir={bidirectional}")
    
    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            h_0: Initial hidden state [num_layers * num_directions, batch, hidden_size]
        
        Returns:
            output: Sequence of hidden states [batch, seq_len, hidden_size * num_directions]
            h_n: Final hidden state [num_layers * num_directions, batch, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
        
        # Process each layer
        layer_input = x
        h_n_list = []
        
        for layer in range(self.num_layers):
            # Forward direction
            fw_idx = layer * self.num_directions
            h_fw = h_0[fw_idx]
            outputs_fw = []
            
            for t in range(seq_len):
                h_fw = self.cells[fw_idx](layer_input[:, t, :], h_fw)
                outputs_fw.append(h_fw)
            
            outputs_fw = torch.stack(outputs_fw, dim=1)
            h_n_list.append(h_fw)
            
            if self.bidirectional:
                # Backward direction
                bw_idx = fw_idx + 1
                h_bw = h_0[bw_idx]
                outputs_bw = []
                
                for t in range(seq_len - 1, -1, -1):
                    h_bw = self.cells[bw_idx](layer_input[:, t, :], h_bw)
                    outputs_bw.insert(0, h_bw)
                
                outputs_bw = torch.stack(outputs_bw, dim=1)
                h_n_list.append(h_bw)
                
                layer_input = torch.cat([outputs_fw, outputs_bw], dim=2)
            else:
                layer_input = outputs_fw
            
            # Apply dropout between layers
            if layer < self.num_layers - 1:
                if not isinstance(layer_input, torch.Tensor):
                    logger.error(f"Dropout Input Type Error: Expected Tensor, got {type(layer_input)}")
                layer_input = self.dropout_layer(layer_input)
        
        h_n = torch.stack(h_n_list, dim=0)
        
        return layer_input, h_n
    
    def get_output_size(self) -> int:
        """Return output feature size."""
        return self.hidden_size * self.num_directions
    
    def get_weight_vector(self) -> np.ndarray:
        """Get all weights as a flat vector for SIAO."""
        weights = []
        for cell in self.cells:
            weights.append(cell.get_weight_vector())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from flat vector."""
        idx = 0
        for cell in self.cells:
            # Compute size via param.numel() — zero extra allocation (avoids NumPy OOM on long runs)
            size = sum(p.numel() for p in cell.parameters())
            cell.set_weight_vector(weights[idx:idx+size], device)
            idx += size
    
    def get_weight_dim(self) -> int:
        """Get total number of weights."""
        return len(self.get_weight_vector())


# =============================================================================
# Enhanced ORNN with Attention
# =============================================================================

class EnhancedORNN(nn.Module):
    """
    Enhanced ORNN with self-attention and layer normalization.
    
    Combines:
    - Bidirectional GRU layers
    - Self-attention mechanism
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        cell_type: str = 'gru',
        dropout: float = 0.1,
        use_attention: bool = True,
        bidirectional: bool = True,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Base ORNN
        self.ornn = ORNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type=cell_type,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        output_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Self-attention (if enabled)
        if use_attention:
            self.attention = SelfAttention(output_size, num_heads=num_heads)
            self.attention_norm = nn.LayerNorm(output_size)
        
        # Projection if using residual
        if input_size != output_size:
            self.input_proj = nn.Linear(input_size, output_size)
        else:
            self.input_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"EnhancedORNN: hidden={hidden_size}, layers={num_layers}, "
                   f"bidir={bidirectional}, attention={use_attention}")
    
    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention and residual connections.
        
        Args:
            x: Input [batch, seq_len, input_size]
            h_0: Optional initial hidden state
            
        Returns:
            output: Sequence output [batch, seq_len, hidden_size * num_directions]
            h_n: Final hidden state
        """
        # Project input for residual connection
        residual = self.input_proj(x)
        
        # ORNN forward
        output, h_n = self.ornn(x, h_0)
        
        # Layer norm + residual
        output = self.layer_norm(output + residual)
        
        # Self-attention
        if self.use_attention:
            attn_out = self.attention(output)
            output = self.attention_norm(output + self.dropout(attn_out))
        
        return output, h_n
    
    def get_output_size(self) -> int:
        """Return output feature size."""
        return self.hidden_size * self.num_directions
    
    def get_weight_vector(self) -> np.ndarray:
        """Get ORNN weights for SIAO optimization."""
        return self.ornn.get_weight_vector()
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set ORNN weights from SIAO."""
        self.ornn.set_weight_vector(weights, device)
    
    def get_weight_dim(self) -> int:
        """Get weight dimension for SIAO."""
        return self.ornn.get_weight_dim()


# =============================================================================
# SIAO-Optimized RNN Trainer
# =============================================================================


class SIAOORNNTrainer:
    """
    Trainer for ORNN with SIAO weight optimization.
    
    Training process:
    1. Use SIAO to find optimal weight initialization
    2. Fine-tune with backpropagation
    """
    
    def __init__(
        self,
        ornn: ORNN,
        output_size: int,
        device: torch.device,
        siao_pop_size: int = 20,
        siao_max_iter: int = 50,
        bp_epochs: int = 100,
        bp_lr: float = 0.001,
        weight_bounds: Tuple[float, float] = (-1.0, 1.0),
        fc_dropout: float = 0.5,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 20
    ):
        """
        Initialize trainer.
        
        Args:
            ornn: ORNN model
            output_size: Number of output classes
            device: GPU/CPU device
            siao_pop_size: SIAO population size
            siao_max_iter: SIAO iterations
            bp_epochs: Backpropagation epochs
            bp_lr: Learning rate for BP
            weight_bounds: Weight bounds for SIAO
        """
        self.ornn = ornn.to(device)
        self.device = device
        self.siao_pop_size = siao_pop_size
        self.siao_max_iter = siao_max_iter
        self.bp_epochs = bp_epochs
        self.bp_lr = bp_lr
        self.weight_bounds = weight_bounds
        
        # Output layer
        self.bp_lr = bp_lr
        self.weight_bounds = weight_bounds
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Output layer with Dropout
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout) if fc_dropout > 0 else nn.Identity(),
            nn.Linear(ornn.get_output_size(), output_size)
        ).to(device)
        
        # Loss function (default, can be overridden)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.siao_history = []
        self.bp_history = []
    
    def _create_objective(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Callable:
        """Create objective function for SIAO."""

        def objective(weights: np.ndarray) -> float:
            # Set weights
            self.ornn.set_weight_vector(weights, self.device)

            # Forward pass
            with torch.no_grad():
                output, _ = self.ornn(X)
                last_hidden = output[:, -1, :]
                logits = self.fc(last_hidden)
                loss = self.criterion(logits, y)

            return loss.item()

        return objective

    def _create_batch_objective(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Callable:
        """Create a batched objective that evaluates multiple weight vectors at once."""

        def batch_objective(weight_vectors: list) -> list:
            """Evaluate a list of weight vectors, return list of losses."""
            losses = []
            with torch.no_grad():
                for weights in weight_vectors:
                    self.ornn.set_weight_vector(weights, self.device)
                    output, _ = self.ornn(X)
                    last_hidden = output[:, -1, :]
                    logits = self.fc(last_hidden)
                    loss = self.criterion(logits, y)
                    losses.append(loss.item())
            return losses

        return batch_objective
    
    def siao_optimize(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> np.ndarray:
        """
        Optimize ORNN weights using SIAO.

        Args:
            X_train: Training data [batch, seq_len, features]
            y_train: Training labels [batch]

        Returns:
            Optimized weight vector
        """
        from src.siao_cnn_ogru.optimizers.siao_optimizer import SelfImprovedAquilaOptimizer

        console.print("[bold cyan]Starting SIAO Weight Optimization for ORNN[/bold cyan]")

        # Get weight dimension
        weight_dim = self.ornn.get_weight_dim()
        console.print(f"[bold]Weight dimension: {weight_dim:,}[/bold]")

        # Create bounds
        lb = self.weight_bounds[0] * np.ones(weight_dim)
        ub = self.weight_bounds[1] * np.ones(weight_dim)

        # Create both single and batch objective
        objective = self._create_objective(X_train, y_train)
        batch_objective = self._create_batch_objective(X_train, y_train)

        # Run SIAO with batch evaluation
        siao = SelfImprovedAquilaOptimizer(
            objective_func=objective,
            dim=weight_dim,
            lb=lb,
            ub=ub,
            pop_size=self.siao_pop_size,
            max_iter=self.siao_max_iter,
            chaos_method='combined',
            minimize=True,
            batch_objective_func=batch_objective,
            convergence_patience=30
        )

        best_weights, best_loss, info = siao.optimize()

        # Set optimized weights
        self.ornn.set_weight_vector(best_weights, self.device)
        self.siao_history = info['history'].tolist()

        console.print(f"[bold green]SIAO optimization complete. Best loss: {best_loss:.4f}[/bold green]")

        return best_weights
    
    def backprop_finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Fine-tune with backpropagation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history dict
        """
        logger.info("=" * 60)
        logger.info("Starting Backpropagation Fine-tuning")
        logger.info("=" * 60)
        
        # Combine parameters
        params = list(self.ornn.parameters()) + list(self.fc.parameters())
        optimizer = optim.Adam(params, lr=self.bp_lr, weight_decay=self.weight_decay)
        # Change to StepLR as per requested specs
        # Scaled step_size to 40 to match research paper's iteration count (125 iters)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.2
        )

        # Mixed precision (AMP) for GPU acceleration
        use_amp = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # Early Stopping Variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        console.print(f"[bold]Starting Backprop finetuning for {self.bp_epochs} epochs...[/bold]")
        if use_amp:
            console.print("[bold cyan]Mixed precision (AMP) enabled[/bold cyan]")

        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("[yellow]{task.fields[metrics]}"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        ) as progress:

            epoch_task = progress.add_task("Training", total=self.bp_epochs, metrics="")

            for epoch in range(self.bp_epochs):
                # Training
                self.ornn.train()
                self.fc.train()

                epoch_loss = 0.0
                correct = 0
                total = 0
                n_batches = len(train_loader)

                batch_task = progress.add_task(
                    f"  Epoch {epoch+1}/{self.bp_epochs}", total=n_batches, metrics=""
                )

                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    # Forward with AMP autocast
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        output, _ = self.ornn(X_batch)
                        last_hidden = output[:, -1, :]
                        logits = self.fc(last_hidden)
                        loss = self.criterion(logits, y_batch)

                    # Backward with scaled gradients
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    _, predicted = logits.max(1)
                    correct += predicted.eq(y_batch).sum().item()
                    total += y_batch.size(0)

                    progress.update(batch_task, advance=1,
                                    metrics=f"loss={loss.item():.4f}")

                # Remove batch bar after epoch completes
                progress.remove_task(batch_task)

                train_loss = epoch_loss / n_batches
                train_acc = correct / total
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)

                # Validation
                val_loss = 0.0
                val_acc = 0.0
                if val_loader:
                    val_loss, val_acc = self._evaluate(val_loader)
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)

                # Scheduler step
                scheduler.step()

                # Update epoch bar with metrics
                progress.update(
                    epoch_task, advance=1,
                    metrics=f"loss={train_loss:.4f} acc={train_acc:.4f} val_acc={val_acc:.4f}"
                )

                # Early Stopping Check
                if self.patience is not None and val_loader:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = {
                            'ornn': self.ornn.state_dict(),
                            'fc': self.fc.state_dict()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            progress.update(epoch_task, description="[bold yellow]Early stopped")
                            if best_model_state:
                                self.ornn.load_state_dict(best_model_state['ornn'])
                                self.fc.load_state_dict(best_model_state['fc'])
                            break
        
        # Restore best model if training completed without early stopping
        if self.patience is not None and val_loader and best_model_state and patience_counter < self.patience:
            self.ornn.load_state_dict(best_model_state['ornn'])
            self.fc.load_state_dict(best_model_state['fc'])
            console.print("[bold green]Restored best model based on validation loss.[/bold green]")

        self.bp_history = history
        return history
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on data loader."""
        self.ornn.eval()
        self.fc.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        use_amp = self.device.type == 'cuda'

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    output, _ = self.ornn(X_batch)
                    last_hidden = output[:, -1, :]
                    logits = self.fc(last_hidden)
                    loss = self.criterion(logits, y_batch)

                total_loss += loss.item()

                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)

        return total_loss / len(data_loader), correct / total
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> Dict:
        """
        Full training pipeline: SIAO + Backprop.
        
        Args:
            X_train: Training data [samples, seq_len, features]
            y_train: Training labels [samples]
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for BP
        
        Returns:
            Training history
        """
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        
        # Step 1: SIAO optimization
        self.siao_optimize(X_train_t, y_train_t)
        
        # Step 2: Backprop fine-tuning
        pin = self.device.type == 'cuda'
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=2, persistent_workers=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                pin_memory=pin, num_workers=2, persistent_workers=True
            )
        
        history = self.backprop_finetune(train_loader, val_loader)
        
        return {'siao': self.siao_history, 'backprop': history}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.ornn.eval()
        self.fc.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output, _ = self.ornn(X_t)
            last_hidden = output[:, -1, :]
            logits = self.fc(last_hidden)
            predictions = logits.argmax(dim=1)
        
        return predictions.cpu().numpy()
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get ORNN embeddings (for downstream tasks)."""
        self.ornn.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output, _ = self.ornn(X_t)
            embeddings = output[:, -1, :]  # Last hidden state
        
        return embeddings.cpu().numpy()


# =============================================================================
# Visualization
# =============================================================================

def plot_ornn_training(history: Dict, save_path: Optional[str] = None):
    """Plot ORNN training history."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # SIAO convergence
        if 'siao' in history and history['siao']:
            axes[0].plot(history['siao'], 'b-', linewidth=2)
            axes[0].set_xlabel('Iteration', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('SIAO Weight Optimization', fontsize=14)
            axes[0].grid(True, alpha=0.3)
        
        # BP loss
        if 'backprop' in history:
            bp = history['backprop']
            axes[1].plot(bp['train_loss'], 'b-', label='Train', linewidth=2)
            if bp['val_loss']:
                axes[1].plot(bp['val_loss'], 'r--', label='Val', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Backprop Fine-tuning Loss', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # BP accuracy
            axes[2].plot(bp['train_acc'], 'b-', label='Train', linewidth=2)
            if bp['val_acc']:
                axes[2].plot(bp['val_acc'], 'r--', label='Val', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Accuracy', fontsize=12)
            axes[2].set_title('Backprop Fine-tuning Accuracy', fontsize=14)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Optimized RNN (ORNN) with SIAO - Demo")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data
    num_samples = 200
    seq_len = 50
    input_size = 43
    num_classes = 6
    
    X = np.random.randn(num_samples, seq_len, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    
    print(f"Data: X={X.shape}, y={y.shape}")
    
    # Create ORNN
    ornn = ORNN(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
        cell_type='gru'
    )
    
    print(f"ORNN weight dimension: {ornn.get_weight_dim()}")
    
    # Create trainer
    trainer = SIAOORNNTrainer(
        ornn=ornn,
        output_size=num_classes,
        device=device,
        siao_pop_size=10,
        siao_max_iter=20,
        bp_epochs=50,
        bp_lr=0.001
    )
    
    # Train
    history = trainer.train(X, y, batch_size=32)
    
    # Evaluate
    predictions = trainer.predict(X)
    accuracy = (predictions == y).mean()
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Plot
    plot_ornn_training(history)
