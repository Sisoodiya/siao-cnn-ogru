"""
Model Enhancement Module for SIAO-CNN-ORNN

Implements accuracy improvement strategies:
1. Class-weighted loss for imbalanced data
2. Larger model architecture with more capacity
3. Time-series data augmentation
4. Ensemble methods for robust predictions

Author: Model Optimization Specialist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List, Dict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Augmentation
# =============================================================================

class TimeSeriesAugmenter:
    """
    Data augmentation for time-series data.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_strength: float = 0.1,
        mixup_alpha: float = 0.2
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_strength = time_warp_strength
        self.mixup_alpha = mixup_alpha
    
    def add_gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = np.random.normal(0, self.noise_std, X.shape)
        return X + noise
    
    def random_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply random scaling to each sample."""
        scales = np.random.uniform(
            self.scale_range[0], self.scale_range[1], 
            size=(X.shape[0], 1, 1)
        )
        return X * scales
    
    def time_warping(self, X: np.ndarray) -> np.ndarray:
        """Apply time warping (slight temporal distortion)."""
        n_samples, time_steps, features = X.shape
        
        warped = np.zeros_like(X)
        for i in range(n_samples):
            # Create warping indices
            warp = np.cumsum(1 + self.time_warp_strength * np.random.randn(time_steps))
            warp = warp / warp[-1] * (time_steps - 1)
            warp = np.clip(warp, 0, time_steps - 1).astype(int)
            
            warped[i] = X[i, warp, :]
        
        return warped
    
    def magnitude_warping(self, X: np.ndarray) -> np.ndarray:
        """Apply smooth magnitude warping."""
        n_samples, time_steps, features = X.shape
        
        # Generate smooth curve
        n_knots = 4
        knots = np.random.normal(1, 0.1, (n_samples, n_knots))
        
        warped = np.zeros_like(X)
        for i in range(n_samples):
            warp_curve = np.interp(
                np.linspace(0, 1, time_steps),
                np.linspace(0, 1, n_knots),
                knots[i]
            )
            warped[i] = X[i] * warp_curve[:, np.newaxis]
        
        return warped
    
    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation - blend two samples.
        
        Note: Returns soft labels for cross-entropy
        """
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(n_samples, 1, 1))
        
        X_mixed = lam * X + (1 - lam) * X[indices]
        
        # For hard labels, use majority
        y_mixed = np.where(lam.squeeze() > 0.5, y, y[indices])
        
        return X_mixed.astype(np.float32), y_mixed
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: List[str] = ['noise', 'scale'],
        augment_ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation and return combined dataset.
        
        Args:
            X: Original data [n_samples, time_steps, features]
            y: Labels [n_samples]
            methods: Augmentation methods to apply
            augment_ratio: Ratio of augmented samples
        
        Returns:
            Augmented X and y
        """
        n_aug = int(len(X) * augment_ratio)
        indices = np.random.choice(len(X), n_aug, replace=True)
        
        X_aug = X[indices].copy()
        y_aug = y[indices].copy()
        
        for method in methods:
            if method == 'noise':
                X_aug = self.add_gaussian_noise(X_aug)
            elif method == 'scale':
                X_aug = self.random_scaling(X_aug)
            elif method == 'warp':
                X_aug = self.time_warping(X_aug)
            elif method == 'magnitude':
                X_aug = self.magnitude_warping(X_aug)
            elif method == 'mixup':
                X_aug, y_aug = self.mixup(X_aug, y_aug)
        
        # Combine original and augmented
        X_combined = np.concatenate([X, X_aug], axis=0)
        y_combined = np.concatenate([y, y_aug], axis=0)
        
        # Shuffle
        perm = np.random.permutation(len(X_combined))
        
        logger.info(f"Augmented: {len(X)} -> {len(X_combined)} samples")
        
        return X_combined[perm], y_combined[perm]


# =============================================================================
# Enhanced CNN Model
# =============================================================================

class EnhancedCNNBlock(nn.Module):
    """Enhanced CNN block with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_residual: bool = True
    ):
        super(EnhancedCNNBlock, self).__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_residual:
            out = out + identity
        
        out = self.relu(out)
        out = self.pool(out)
        
        return out


class EnhancedCNN(nn.Module):
    """Enhanced CNN with larger capacity and residual connections."""
    
    def __init__(
        self,
        input_time_steps: int = 50,
        input_features: int = 43,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        embedding_dim: int = 512,
        dropout: float = 0.4
    ):
        super(EnhancedCNN, self).__init__()
        
        self.input_channels = 1
        
        # Build layers
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(EnhancedCNNBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flatten size
        self._flat_size = self._get_flat_size(input_time_steps, input_features)
        
        # FC head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _get_flat_size(self, time_steps: int, features: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, time_steps, features)
            out = self.conv_layers(dummy)
            return out.numel()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = self.fc(x)
        
        return x


# =============================================================================
# Enhanced RNN
# =============================================================================

class EnhancedGRU(nn.Module):
    """Enhanced GRU with attention mechanism."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super(EnhancedGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU forward
        output, h_n = self.gru(x)  # output: [batch, seq, hidden*directions]
        
        if self.use_attention and x.size(1) > 1:
            # Attention mechanism
            attn_weights = self.attention(output)  # [batch, seq, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * output, dim=1)  # [batch, hidden*directions]
            return context
        else:
            # Use last hidden state
            return output[:, -1, :]
    
    def get_output_size(self) -> int:
        return self.hidden_size * self.num_directions


# =============================================================================
# Enhanced SIAO-CNN-ORNN Model
# =============================================================================

class EnhancedSIAOCNNORNN(nn.Module):
    """
    Enhanced SIAO-CNN-ORNN with larger capacity.
    """
    
    def __init__(
        self,
        input_time_steps: int = 50,
        input_features: int = 43,
        cnn_channels: Tuple[int, ...] = (64, 128, 256),
        cnn_embedding_dim: int = 512,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.4,
        use_attention: bool = True
    ):
        super(EnhancedSIAOCNNORNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Enhanced CNN
        self.cnn = EnhancedCNN(
            input_time_steps=input_time_steps,
            input_features=input_features,
            channels=cnn_channels,
            embedding_dim=cnn_embedding_dim,
            dropout=dropout
        )
        
        # Enhanced GRU with attention
        self.rnn = EnhancedGRU(
            input_size=cnn_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Classifier head
        rnn_out_size = self.rnn.get_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        logger.info(f"Enhanced Model: CNN channels={cnn_channels}, "
                   f"RNN hidden={rnn_hidden_size}, attention={use_attention}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features
        cnn_out = self.cnn(x)  # [batch, embedding]
        
        # Reshape for RNN
        rnn_in = cnn_out.unsqueeze(1)  # [batch, seq=1, embedding]
        
        # RNN + attention
        rnn_out = self.rnn(rnn_in)  # [batch, hidden*2]
        
        # Classify
        logits = self.classifier(rnn_out)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)


# =============================================================================
# Ensemble Model
# =============================================================================

class EnsembleModel:
    """
    Ensemble of multiple SIAO-CNN-ORNN models.
    
    Uses bagging with different:
    - Random seeds
    - Data subsets
    - Slight architecture variations
    """
    
    def __init__(
        self,
        n_models: int = 3,
        input_time_steps: int = 50,
        input_features: int = 43,
        num_classes: int = 6,
        device: torch.device = None
    ):
        self.n_models = n_models
        self.input_time_steps = input_time_steps
        self.input_features = input_features
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = []
        self.histories = []
        
        logger.info(f"Ensemble: {n_models} models on {self.device}")
    
    def _create_model(self, variant: int) -> EnhancedSIAOCNNORNN:
        """Create model with slight variations."""
        base_channels = [(64, 128, 256), (32, 64, 128, 256), (64, 128, 256, 512)]
        hidden_sizes = [256, 192, 320]
        
        return EnhancedSIAOCNNORNN(
            input_time_steps=self.input_time_steps,
            input_features=self.input_features,
            cnn_channels=base_channels[variant % len(base_channels)],
            cnn_embedding_dim=512,
            rnn_hidden_size=hidden_sizes[variant % len(hidden_sizes)],
            num_classes=self.num_classes,
            dropout=0.3 + 0.1 * (variant % 2)
        ).to(self.device)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_augmentation: bool = True,
        use_class_weights: bool = True
    ) -> List[Dict]:
        """
        Train ensemble of models.
        """
        logger.info("=" * 60)
        logger.info("Training Ensemble Model")
        logger.info("=" * 60)
        
        # Compute class weights
        class_weights = None
        if use_class_weights:
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            logger.info(f"Class weights: {weights}")
        
        # Augmenter
        augmenter = TimeSeriesAugmenter()
        
        for i in range(self.n_models):
            logger.info(f"\n--- Training Model {i+1}/{self.n_models} ---")
            
            # Set different seed
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            # Create model
            model = self._create_model(i)
            
            # Augment data (different augmentation per model)
            if use_augmentation:
                methods = [['noise', 'scale'], ['noise', 'warp'], ['scale', 'magnitude']][i % 3]
                X_train_aug, y_train_aug = augmenter.augment(X_train, y_train, methods=methods)
            else:
                X_train_aug, y_train_aug = X_train, y_train
            
            # Train
            history = self._train_single_model(
                model, X_train_aug, y_train_aug, X_val, y_val,
                epochs, batch_size, class_weights
            )
            
            self.models.append(model)
            self.histories.append(history)
        
        # Final ensemble evaluation
        val_acc = self.evaluate(X_val, y_val)
        logger.info(f"\nEnsemble Validation Accuracy: {val_acc:.4f}")
        
        return self.histories
    
    def _train_single_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        class_weights: Optional[torch.Tensor]
    ) -> Dict:
        """Train a single model."""
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
            
            scheduler.step()
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation
            val_loss, val_acc = self._evaluate_single(model, X_val, y_val, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: Train_Acc={train_acc:.4f}, Val_Acc={val_acc:.4f}")
        
        # Load best
        if best_state:
            model.load_state_dict(best_state)
        
        logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
        
        return history
    
    def _evaluate_single(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate single model."""
        model.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = model(X_t)
            loss = criterion(logits, y_t).item()
            preds = logits.argmax(dim=1)
            acc = (preds == y_t).float().mean().item()
        
        return loss, acc
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using voting."""
        all_preds = []
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds = model.predict(X_t).cpu().numpy()
            all_preds.append(preds)
        
        # Voting
        all_preds = np.array(all_preds)  # [n_models, n_samples]
        
        # Majority vote
        from scipy import stats
        ensemble_preds = stats.mode(all_preds, axis=0, keepdims=False)[0]
        
        return ensemble_preds.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average probability prediction."""
        all_probas = []
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X_t)
                proba = F.softmax(logits, dim=1).cpu().numpy()
            all_probas.append(proba)
        
        # Average probabilities
        return np.mean(all_probas, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate ensemble accuracy."""
        preds = self.predict(X)
        return accuracy_score(y, preds)
    
    def get_classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """Get classification report."""
        preds = self.predict(X)
        return classification_report(
            y, preds,
            target_names=['Steady', 'Transient', 'PORV', 'SGTR', 'FWLB', 'RCP'],
            digits=4
        )


# =============================================================================
# Training Function
# =============================================================================

def train_enhanced_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'ensemble',  # 'single' or 'ensemble'
    epochs: int = 100,
    batch_size: int = 32,
    use_augmentation: bool = True,
    use_class_weights: bool = True,
    n_ensemble_models: int = 3
) -> Tuple:
    """
    Train enhanced model with all optimizations.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: 'single' for enhanced single model, 'ensemble' for ensemble
        epochs: Training epochs
        batch_size: Batch size
        use_augmentation: Whether to use data augmentation
        use_class_weights: Whether to use class-weighted loss
        n_ensemble_models: Number of models in ensemble
    
    Returns:
        Trained model/ensemble and history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Data: Train={X_train.shape}, Val={X_val.shape}")
    
    if model_type == 'ensemble':
        ensemble = EnsembleModel(
            n_models=n_ensemble_models,
            input_time_steps=X_train.shape[1],
            input_features=X_train.shape[2],
            device=device
        )
        
        histories = ensemble.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            use_class_weights=use_class_weights
        )
        
        print("\n" + "=" * 60)
        print("ENSEMBLE CLASSIFICATION REPORT")
        print("=" * 60)
        print(ensemble.get_classification_report(X_val, y_val))
        
        return ensemble, histories
    
    else:
        # Single enhanced model
        model = EnhancedSIAOCNNORNN(
            input_time_steps=X_train.shape[1],
            input_features=X_train.shape[2],
            cnn_channels=(64, 128, 256),
            cnn_embedding_dim=512,
            rnn_hidden_size=256,
            num_classes=len(np.unique(y_train))
        ).to(device)
        
        # Augment if requested
        if use_augmentation:
            augmenter = TimeSeriesAugmenter()
            X_train, y_train = augmenter.augment(X_train, y_train, methods=['noise', 'scale'])
        
        # Class weights
        class_weights = None
        if use_class_weights:
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
            
            scheduler.step()
            
            history['train_loss'].append(total_loss / len(train_loader))
            history['train_acc'].append(correct / total)
            
            # Validation
            model.eval()
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model(X_val_t)
                val_preds = logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train_Acc={history['train_acc'][-1]:.4f}, Val_Acc={val_acc:.4f}")
        
        return model, history


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Model Enhancement Module - Demo")
    print("=" * 60)
    
    # Dummy data
    X_train = np.random.randn(400, 50, 43).astype(np.float32)
    y_train = np.random.randint(0, 6, 400)
    X_val = np.random.randn(100, 50, 43).astype(np.float32)
    y_val = np.random.randint(0, 6, 100)
    
    # Train ensemble
    ensemble, histories = train_enhanced_model(
        X_train, y_train, X_val, y_val,
        model_type='ensemble',
        epochs=30,
        n_ensemble_models=3
    )
    
    print(f"\nFinal Ensemble Accuracy: {ensemble.evaluate(X_val, y_val):.4f}")
