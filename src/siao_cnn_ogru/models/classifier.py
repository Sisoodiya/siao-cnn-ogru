"""
SIAO-CNN-ORNN Complete Classifier

End-to-end trainable model for IP-200 reactor fault classification
using Self-Improved Aquila Optimizer enhanced CNN-RNN architecture.

Classes:
0 → Steady State
1 → Transient (Power Change)
2 → PORV Stuck Open
3 → SGTR (Steam Generator Tube Rupture)
4 → FWLB (Feedwater Line Break)
5 → RCP Failure

Author: Classification Head Designer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Class Mapping
# =============================================================================

CLASS_NAMES = {
    0: 'Steady State',
    1: 'Transient',
    2: 'PORV',
    3: 'SGTR',
    4: 'FWLB',
    5: 'RCP Failure'
}

CLASS_LABELS = list(CLASS_NAMES.keys())


# =============================================================================
# Classification Head
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Fully Connected classification head with softmax output.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5
    ):
        super(ClassificationHead, self).__init__()
        
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [128]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.fc = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            Logits [batch, num_classes]
        """
        return self.fc(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities (softmax)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# =============================================================================
# Complete SIAO-CNN-ORNN Model
# =============================================================================

class SIAOCNNORNN(nn.Module):
    """
    Complete SIAO-CNN-ORNN Classifier.
    
    Architecture:
    1. CNN: Extract spatial features from time windows
    2. ORNN: Process temporal dependencies (SIAO-optimized)
    3. FC Head: Classification with softmax
    
    Input: [batch, time_steps, features]
    Output: [batch, num_classes]
    """
    
    def __init__(
        self,
        input_time_steps: int = 50,
        input_features: int = 43,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        cnn_embedding_dim: int = 256,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_cell_type: str = 'gru',
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super(SIAOCNNORNN, self).__init__()
        
        self.input_time_steps = input_time_steps
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Import local modules
        from cnn_model import CNNFeatureExtractor
        from ornn_model import ORNN
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            input_channels=1,
            input_time_steps=input_time_steps,
            input_features=input_features,
            conv_channels=cnn_channels,
            embedding_dim=cnn_embedding_dim,
            dropout=dropout
        )
        
        # ORNN (SIAO-optimized RNN)
        self.ornn = ORNN(
            input_size=cnn_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            cell_type=rnn_cell_type,
            dropout=dropout
        )
        
        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=rnn_hidden_size,
            num_classes=num_classes,
            hidden_dims=[64],
            dropout=dropout
        )
        
        logger.info(f"SIAO-CNN-ORNN initialized:")
        logger.info(f"  Input: [{input_time_steps}, {input_features}]")
        logger.info(f"  CNN embedding: {cnn_embedding_dim}")
        logger.info(f"  RNN hidden: {rnn_hidden_size}")
        logger.info(f"  Classes: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, time_steps, features]
        
        Returns:
            Logits [batch, num_classes]
        """
        # CNN features
        cnn_out = self.cnn(x)  # [batch, cnn_embedding_dim]
        
        # Reshape for RNN: [batch, seq_len=1, features]
        rnn_in = cnn_out.unsqueeze(1)
        
        # ORNN processing
        rnn_out, _ = self.ornn(rnn_in)  # [batch, 1, rnn_hidden]
        rnn_out = rnn_out.squeeze(1)  # [batch, rnn_hidden]
        
        # Classification
        logits = self.classifier(rnn_out)  # [batch, num_classes]
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get CNN-ORNN embeddings before classification."""
        cnn_out = self.cnn(x)
        rnn_in = cnn_out.unsqueeze(1)
        rnn_out, _ = self.ornn(rnn_in)
        return rnn_out.squeeze(1)


# =============================================================================
# Trainer
# =============================================================================

class SIAOCNNORNNTrainer:
    """
    Trainer for complete SIAO-CNN-ORNN model.
    """
    
    def __init__(
        self,
        model: SIAOCNNORNN,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on data loader."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        return total_loss / len(data_loader), correct / total
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping: int = 20
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            early_stopping: Patience for early stopping
        
        Returns:
            Training history
        """
        logger.info("=" * 60)
        logger.info("Starting SIAO-CNN-ORNN Training")
        logger.info("=" * 60)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}"
                if val_loader:
                    msg += f", Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}"
                logger.info(msg)
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            logger.info(f"Loaded best model with Val_Acc={best_val_acc:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(
                    X[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=self.device
                )
                preds = self.model.predict(batch)
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get class probabilities."""
        self.model.eval()
        
        probas = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(
                    X[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=self.device
                )
                proba = self.model.predict_proba(batch)
                probas.append(proba.cpu().numpy())
        
        return np.concatenate(probas)
    
    def get_classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> str:
        """Generate classification report."""
        predictions = self.predict(X)
        
        return classification_report(
            y, predictions,
            target_names=[CLASS_NAMES[i] for i in range(6)],
            digits=4
        )
    
    def get_confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Generate confusion matrix."""
        predictions = self.predict(X)
        return confusion_matrix(y, predictions)


# =============================================================================
# Visualization
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        if history['val_loss']:
            axes[0].plot(history['val_loss'], 'r--', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['train_acc'], 'b-', label='Train', linewidth=2)
        if history['val_acc']:
            axes[1].plot(history['val_acc'], 'r--', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[CLASS_NAMES[i] for i in range(6)],
            yticklabels=[CLASS_NAMES[i] for i in range(6)]
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix - SIAO-CNN-ORNN', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib/seaborn not available")


# =============================================================================
# Quick Training Function
# =============================================================================

def train_siao_cnn_ornn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[SIAOCNNORNNTrainer, Dict]:
    """
    Quick function to train SIAO-CNN-ORNN.
    
    Args:
        X_train: Training data [samples, time_steps, features]
        y_train: Training labels [samples]
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Trained trainer and history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Get input shape
    _, time_steps, features = X_train.shape
    num_classes = len(np.unique(y_train))
    
    # Create model
    model = SIAOCNNORNN(
        input_time_steps=time_steps,
        input_features=features,
        num_classes=num_classes
    )
    
    # Create trainer
    trainer = SIAOCNNORNNTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    return trainer, history


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("SIAO-CNN-ORNN Complete Classifier - Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data
    X_train = np.random.randn(400, 50, 43).astype(np.float32)
    y_train = np.random.randint(0, 6, size=400)
    X_val = np.random.randn(100, 50, 43).astype(np.float32)
    y_val = np.random.randint(0, 6, size=100)
    
    # Train
    trainer, history = train_siao_cnn_ornn(
        X_train, y_train, X_val, y_val,
        epochs=50, batch_size=32
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Classification Report:")
    print(trainer.get_classification_report(X_val, y_val))
    
    # Plot
    plot_training_history(history)
    
    cm = trainer.get_confusion_matrix(X_val, y_val)
    plot_confusion_matrix(cm)
