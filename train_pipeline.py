"""
Complete SIAO-CNN-OGRU Training Pipeline

Integrates all components:
1. Data Pipeline - Load and preprocess reactor data
2. Window Processor - Create sliding windows
3. CNN Feature Extractor - Extract spatial features
4. ORNN - SIAO-optimized RNN for classification

Author: SIAO-CNN-OGRU Integration
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Sequence, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.siao_cnn_ogru.data.class_metadata import RESEARCH_CLASS_CODES_14, build_label_metadata_map


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible CV behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_fold_inputs(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit scaler on train fold only, then transform train/val."""
    _, _, n_features = X_train.shape
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    X_train_norm = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_val_norm = scaler.transform(X_val_2d).reshape(X_val.shape).astype(np.float32)
    return X_train_norm, X_val_norm


def _adaptive_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    target_percentile: int = 50,
    min_samples_per_class: int = 6
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply SMOTE for minority classes with enough support.

    Avoids hard-coded class IDs and gracefully handles subsets (e.g., 13 classes).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_train, {"applied": False, "reason": "imblearn_not_installed"}

    class_counts = np.bincount(y_train, minlength=num_classes)
    active_counts = class_counts[class_counts > 0]

    if active_counts.size == 0:
        return X_train, y_train, {"applied": False, "reason": "no_active_classes"}

    target_count = int(np.percentile(active_counts, target_percentile))
    sampling_strategy = {
        cls_idx: target_count
        for cls_idx, count in enumerate(class_counts)
        if min_samples_per_class <= count < target_count
    }

    if not sampling_strategy:
        return X_train, y_train, {
            "applied": False,
            "reason": "no_eligible_minority_class",
            "target_count": target_count
        }

    min_eligible = min(class_counts[cls_idx] for cls_idx in sampling_strategy.keys())
    k_neighbors = max(1, min(5, min_eligible - 1))

    n_samples, time_steps, n_features = X_train.shape
    X_flat = X_train.reshape(n_samples, time_steps * n_features)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X_flat, y_train)
    X_res = X_res.reshape(-1, time_steps, n_features).astype(np.float32)

    return X_res, y_res, {
        "applied": True,
        "target_count": target_count,
        "k_neighbors": k_neighbors,
        "classes": sorted(list(sampling_strategy.keys()))
    }


def run_complete_pipeline(
    data_dir: str = 'data/Operation_csv_data/',
    window_size: int = 100,
    stride: int = 25,
    cnn_embedding_dim: int = 512,  # Increased for larger feature space (97 features)
    wks_dim: int = 97,  # Legacy arg; WKS feature dimension is inferred from data.
    rnn_hidden_size: int = 224,  # Optuna optimized (was 128)
    rnn_num_layers: int = 2,  # Optuna optimized (was 1)
    num_classes: Optional[int] = None,  # Optional override; auto-detected from active class metadata.
    active_class_codes: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    wks_pop_size: int = 15,
    wks_max_iter: int = 30,
    siao_pop_size: int = 30,   # Phase 1: bidir doubled dim to 2.67M; 30 agents fits 4GB VRAM (30x100=3000 evals vs old 25x40=1000)
    siao_max_iter: int = 100,  # Phase 1: increased 40→100
    bp_epochs: int = 150,       # Phase 1: increased 100→150 (early stopping still guards)
    bp_lr: float = 0.00157,  # Optuna optimized (was 0.001)
    fc_dropout: float = 0.164,  # Optuna optimized (was 0.2)
    weight_decay: float = 1.97e-05,  # Optuna optimized (was 1e-5)
    batch_size: int = 163,
    use_class_weights: bool = True,
    label_smoothing: float = 0.05,
    use_smote: bool = True,
    balance_to_max: bool = False,
    smote_target_percentile: int = 50,
    smote_min_samples: int = 6,
    n_folds: int = 10,  # Phase 1: increased 5→10 (+12% training data per fold)
    random_seed: int = 42,
    use_cache: bool = True,
    cache_dir: str = 'data/processed_raw',
    save_dir: str = 'results/'  # Results directory
) -> Dict:
    """
    Run the complete SIAO-CNN-OGRU training pipeline.
    
    Args:
        data_dir: Path to data directory
        window_size: Sliding window size
        stride: Window stride
        cnn_embedding_dim: CNN output dimension
        rnn_hidden_size: RNN hidden size
        num_classes: Number of output classes
        test_size: Validation split ratio
        siao_pop_size: SIAO population size
        siao_max_iter: SIAO iterations
        bp_epochs: Backpropagation epochs
        batch_size: Training batch size
    
    Returns:
        Dictionary with results and history
    """
    
    # Rich Imports
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    
    console = Console()
    _set_seed(random_seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    console.print(f"[bold blue]Device:[/bold blue] {device}")
    
    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    console.print(Panel("[bold green]Step 1: Loading and Preprocessing Data[/bold green]", box=box.DOUBLE))
    
    from src.siao_cnn_ogru.data.nppad_loader import NPPADDataPipeline
    
    selected_class_codes = tuple(active_class_codes) if active_class_codes is not None else RESEARCH_CLASS_CODES_14

    with console.status("[bold green]Ingesting NPPAD data...[/bold green]", spinner="dots"):
        pipeline = NPPADDataPipeline(
            data_dir=data_dir,
            max_timesteps=window_size,  # Match window size
            normalization='none',
            handle_missing='interpolate',
            active_class_codes=selected_class_codes,
        )

        # Use caching to speed up subsequent runs
        X_raw, y_raw = pipeline.run(use_cache=use_cache, cache_dir=cache_dir)

    class_codes = [pipeline.label_to_class[idx] for idx in sorted(pipeline.label_to_class)]
    num_classes_detected = len(class_codes)
    if num_classes is not None and num_classes != num_classes_detected:
        console.print(
            f" [yellow]num_classes={num_classes} overridden by detected active class count {num_classes_detected}.[/yellow]"
        )
    num_classes = num_classes_detected
    class_metadata_map = build_label_metadata_map(class_codes)

    console.print(f" [bold]Raw data loaded:[/bold] X={X_raw.shape}, y={y_raw.shape}")
    console.print(f" [bold]Classes detected:[/bold] {num_classes} -> {class_codes}")
    
    # =========================================================================
    # Step 2: Create Sliding Windows
    # =========================================================================
    console.print(Panel("[bold green]Step 2: Creating Sliding Windows[/bold green]", box=box.DOUBLE))
    
    from src.siao_cnn_ogru.data.window_processor import SlidingWindowProcessor
    
    with console.status("[bold green]Processing windows...[/bold green]", spinner="dots"):
        window_proc = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            padding='zero'
        )
        
        X_windows, y_windows, window_group_ids = window_proc.transform(
            X_raw, y_raw, return_sample_indices=True
        )
    
    console.print(f" [bold]Windowed data:[/bold] X={X_windows.shape}, y={y_windows.shape}")
    console.print(
        f" [bold]Group IDs:[/bold] {len(np.unique(window_group_ids))} source samples for grouped CV"
    )
    
    # =========================================================================
    # Step 3: 5-Fold Cross-Validation Setup
    # =========================================================================
    console.print(Panel("[bold green]Step 3: Cross-Validation Setup[/bold green]", box=box.DOUBLE))
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from src.siao_cnn_ogru.visualization import plot_training_results, plot_confusion_matrix_heatmap

    class_counts_overall = np.bincount(y_windows, minlength=num_classes)
    active_classes = np.where(class_counts_overall > 0)[0]
    min_class_count = int(class_counts_overall[active_classes].min())
    min_group_count_per_class = int(min(len(np.unique(window_group_ids[y_windows == cls])) for cls in active_classes))
    effective_folds = min(n_folds, min_class_count, min_group_count_per_class)
    if effective_folds < 2:
        raise ValueError(
            f"Not enough samples for cross-validation: minimum class count is {min_class_count}. "
            "Need at least 2 samples in each active class."
        )
    if effective_folds != n_folds:
        console.print(
            f" [yellow]Requested {n_folds} folds, but smallest class has {min_class_count} windows "
            f"and {min_group_count_per_class} source groups. Using {effective_folds} folds instead.[/yellow]"
        )

    # Initialize grouped stratified CV to prevent same-source windows appearing in both train/val
    split_strategy = "StratifiedGroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=random_seed)
        split_iterator = splitter.split(X_windows, y_windows, groups=window_group_ids)
    except Exception:
        split_strategy = "StratifiedKFold (fallback)"
        console.print(
            " [yellow]StratifiedGroupKFold unavailable. Falling back to StratifiedKFold; "
            "group isolation is disabled.[/yellow]"
        )
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_seed)
        split_iterator = splitter.split(X_windows, y_windows)

    console.print(f" [bold]CV splitter:[/bold] {split_strategy}")
    
    fold_accuracies = []
    fold_macro_f1 = []
    oof_y_true = []
    oof_y_pred = []
    
    # =========================================================================
    # Step 4: Cross-Validation Loop
    # =========================================================================
    
    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        console.print(f"\n[bold magenta]=== Fold {fold+1}/{effective_folds} ===[/bold magenta]")
        
        X_train, X_val = X_windows[train_idx], X_windows[val_idx]
        y_train, y_val = y_windows[train_idx], y_windows[val_idx]

        # Fold-safe normalization (prevents leakage from validation fold)
        X_train, X_val = _normalize_fold_inputs(X_train, X_val)

        # Optional adaptive SMOTE for supported minority classes
        if use_smote:
            smote_percentile = 100 if balance_to_max else smote_target_percentile
            X_train, y_train, smote_info = _adaptive_smote(
                X_train,
                y_train,
                num_classes=num_classes,
                target_percentile=smote_percentile,
                min_samples_per_class=smote_min_samples
            )
            if smote_info["applied"]:
                console.print(
                    f" [yellow]SMOTE applied: classes {smote_info['classes']} "
                    f"-> target {smote_info['target_count']} (k={smote_info['k_neighbors']})[/yellow]"
                )
            else:
                console.print(
                    f" [green]SMOTE skipped: {smote_info['reason']}[/green]"
                )

        console.print(f" [bold]Train:[/bold] {len(y_train)} samples, [bold]Val:[/bold] {len(y_val)} samples")

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        
        # ---------------------------------------------------------------------
        # CNN Feature Extraction (Pre-trained for each fold)
        # ---------------------------------------------------------------------
        input_channels = X_train.shape[2]

        from src.siao_cnn_ogru.models.cnn_model import create_cnn_extractor
        cnn = create_cnn_extractor(
            input_shape=(window_size, input_channels),
            embedding_dim=cnn_embedding_dim,
            dropout=0.2
        ).to(device)

        # --- Pre-train CNN with a temporary classification head ---
        cnn_head = nn.Linear(cnn_embedding_dim, num_classes).to(device)
        cnn_criterion = nn.CrossEntropyLoss()
        cnn_optimizer = torch.optim.Adam(
            list(cnn.parameters()) + list(cnn_head.parameters()),
            lr=0.001, weight_decay=1e-4
        )
        cnn_epochs = 30
        use_amp = device.type == 'cuda'
        cnn_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        console.print(f"  [bold]Fold {fold+1}: Pre-training CNN ({cnn_epochs} epochs)...[/bold]")

        cnn_dataset = torch.utils.data.TensorDataset(X_train_t, torch.tensor(y_train, dtype=torch.long).to(device))
        cnn_loader = torch.utils.data.DataLoader(cnn_dataset, batch_size=256, shuffle=True)

        cnn.train()
        cnn_head.train()
        for cnn_ep in range(cnn_epochs):
            for X_cnn_batch, y_cnn_batch in cnn_loader:
                cnn_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    emb = cnn(X_cnn_batch.unsqueeze(1))
                    logits = cnn_head(emb)
                    loss = cnn_criterion(logits, y_cnn_batch)
                cnn_scaler.scale(loss).backward()
                cnn_scaler.step(cnn_optimizer)
                cnn_scaler.update()

        console.print(f"  [green]CNN pre-training complete (final loss: {loss.item():.4f})[/green]")
        del cnn_head, cnn_optimizer, cnn_criterion, cnn_scaler, cnn_dataset, cnn_loader

        def extract_cnn_features_batch(model, X_tensor, batch_size=256):
            model.eval()
            embeddings = []
            use_amp = device.type == 'cuda'
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size].unsqueeze(1)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        emb = model(batch)
                    embeddings.append(emb.cpu().numpy())
            return np.vstack(embeddings)

        with console.status(f"[bold]Fold {fold+1}: Extracting CNN features...[/bold]", spinner="dots"):
            X_train_cnn = extract_cnn_features_batch(cnn, X_train_t)
            X_val_cnn = extract_cnn_features_batch(cnn, X_val_t)
            
        # ---------------------------------------------------------------------
        # Statistical & WKS Features
        # ---------------------------------------------------------------------
        from src.siao_cnn_ogru.optimizers.aquila_optimizer import WKSOptimizer
        from src.siao_cnn_ogru.features.feature_extractor import extract_statistical_features
        
        with console.status(f"[bold]Fold {fold+1}: Extracting Statistical features...[/bold]", spinner="dots"):
            X_train_stat = extract_statistical_features(X_train)
            X_val_stat = extract_statistical_features(X_val)

        wks_opt = WKSOptimizer(pop_size=wks_pop_size, max_iter=wks_max_iter)
        
        with console.status(f"[bold]Fold {fold+1}: Optimizing WKS parameters...[/bold]", spinner="simpleDotsScrolling"):
            # Suppress logging inside the status context if possible by reducing log level temporarily or just rely on console
            optimal_omega, _, _ = wks_opt.optimize(X_train, y_train)
            
        X_train_wks = wks_opt.extract_wks_features(X_train, omega=optimal_omega)
        X_val_wks = wks_opt.extract_wks_features(X_val, omega=optimal_omega)
        
        # Combine Features
        X_train_combined = np.hstack([X_train_cnn, X_train_stat, X_train_wks])
        X_val_combined = np.hstack([X_val_cnn, X_val_stat, X_val_wks])
        
        # ---------------------------------------------------------------------
        # ORNN Training
        # ---------------------------------------------------------------------
        from src.siao_cnn_ogru.models.ornn_model import ORNN, SIAOORNNTrainer
        
        combined_input_size = X_train_combined.shape[1]
        
        ornn = ORNN(
            input_size=combined_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            cell_type='gru',
            bidirectional=True  # Phase 1: bidir GRU — sees both past + future context
        )
        
        trainer = SIAOORNNTrainer(
            ornn=ornn,
            output_size=num_classes,
            device=device,
            siao_pop_size=siao_pop_size,
            siao_max_iter=siao_max_iter,
            bp_epochs=bp_epochs,
            bp_lr=bp_lr,  # Now uses function parameter
            weight_bounds=(-1.0, 1.0),
            fc_dropout=fc_dropout,  # Now uses function parameter
            weight_decay=weight_decay,  # Now uses function parameter
            patience=20
        )
        
        # Class Weights
        if use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes)
            # Add 1 to avoid division by zero if a class is missing in a fold (unlikely with StratifiedKFold)
            weights = len(y_train) / (num_classes * (class_counts + 1))
            weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
            trainer.criterion = nn.CrossEntropyLoss(
                weight=weights_t,
                label_smoothing=label_smoothing
            )
        else:
            trainer.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
        # Prepare Data for RNN (Numpy format for trainer.train)
        # Ensure 3D shape: [samples, 1, features]
        if X_train_combined.ndim == 2:
            X_train_rnn_np = np.expand_dims(X_train_combined, axis=1)
            X_val_rnn_np = np.expand_dims(X_val_combined, axis=1)
        else:
            X_train_rnn_np = X_train_combined
            X_val_rnn_np = X_val_combined
            
        # Used for evaluation later
        X_train_rnn = torch.tensor(X_train_rnn_np, dtype=torch.float32).to(device)
        X_val_rnn = torch.tensor(X_val_rnn_np, dtype=torch.float32).to(device)
        
        # Train - Pass numpy arrays as expected by SIAOORNNTrainer
        # It handles tensor conversion and dataloader creation internally
        result_dict = trainer.train(
            X_train_rnn_np, y_train,
            X_val_rnn_np, y_val,
            batch_size=batch_size
        )
        history = result_dict['backprop']
        
        # Plot training results
        plot_training_results(history, fold_idx=fold, save_dir='results/plots')
        
        # Evaluate Fold
        ornn.eval()
        trainer.fc.eval()
        with torch.no_grad():
            ornn_out, _ = ornn(X_val_rnn)
            last_hidden = ornn_out[:, -1, :]
            outputs = trainer.fc(last_hidden)
            _, preds = torch.max(outputs, 1)
            
            # Plot confusion matrix
            y_pred_np = preds.cpu().numpy()

            plot_confusion_matrix_heatmap(
                y_val, y_pred_np, 
                classes=class_codes,
                fold_idx=fold,
                save_dir='results/plots'
            )
            
            fold_acc = accuracy_score(y_val, y_pred_np)
            fold_accuracies.append(fold_acc)
            from sklearn.metrics import f1_score
            fold_f1 = f1_score(y_val, y_pred_np, average='macro')
            fold_macro_f1.append(fold_f1)
            oof_y_true.append(y_val.copy())
            oof_y_pred.append(y_pred_np.copy())
            
        console.print(
            f"[bold green]Fold {fold+1} Accuracy: {fold_acc*100:.2f}% | Macro-F1: {fold_f1*100:.2f}%[/bold green]"
        )
        
    # =========================================================================
    # Final Results Aggregation
    # =========================================================================
    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    avg_f1 = np.mean(fold_macro_f1) if fold_macro_f1 else 0.0
    
    console.print(Panel(f"[bold]{effective_folds}-Fold Cross-Validation Results[/bold]\n"
                        f"\nFold Scores: {[f'{x*100:.2f}%' for x in fold_accuracies]}\n"
                        f"\n[bold green]Average Accuracy: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%)[/bold green]\n"
                        f"[bold cyan]Average Macro-F1: {avg_f1*100:.2f}%[/bold cyan]\n"
                        f"Target: 98.74%", title="Final Report", box=box.DOUBLE))

    oof_true = np.concatenate(oof_y_true) if oof_y_true else np.array([], dtype=np.int64)
    oof_pred = np.concatenate(oof_y_pred) if oof_y_pred else np.array([], dtype=np.int64)

    return {
        'avg_accuracy': avg_acc,
        'fold_accuracies': fold_accuracies,
        'std_accuracy': std_acc,
        'avg_macro_f1': avg_f1,
        'fold_macro_f1': fold_macro_f1,
        'oof_y_true': oof_true,
        'oof_y_pred': oof_pred,
        'class_codes': class_codes,
        'cv_splitter': split_strategy,
        'class_metadata': [
            {
                'label': idx,
                'code': class_metadata_map[idx].code,
                'full_name': class_metadata_map[idx].full_name,
                'source': class_metadata_map[idx].source,
            }
            for idx in range(num_classes)
        ],
    }


# =============================================================================
# Quick Start Function
# =============================================================================

def quick_start():
    """
    Quick start with NPPAD dataset and optimized parameters.
    
    Usage in Colab:
        from train_pipeline import quick_start
        results = quick_start()
    """
    return run_complete_pipeline(
        data_dir='data/Operation_csv_data/',
        window_size=100,
        stride=25,
        cnn_embedding_dim=256,  # Optuna best; better bias-variance balance for subset runs
        wks_dim=97,  # NPPAD has 97 features
        rnn_hidden_size=224,  # Optuna optimized
        rnn_num_layers=2,  # Optuna optimized
        active_class_codes=RESEARCH_CLASS_CODES_14,
        test_size=0.2,
        wks_pop_size=15,
        wks_max_iter=30,
        siao_pop_size=25,  # Optuna optimized
        siao_max_iter=40,  # Optuna optimized
        bp_epochs=100,
        bp_lr=0.00157,  # Optuna optimized
        fc_dropout=0.164,  # Optuna optimized
        weight_decay=1.97e-05,  # Optuna optimized
        batch_size=163
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SIAO-CNN-OGRU Training Pipeline for NPPAD Dataset'
    )
    parser.add_argument(
        '--folds', type=int, default=10,
        help='Number of cross-validation folds (default: 10)'
    )
    parser.add_argument(
        '--epochs', type=int, default=150,
        help='Number of training epochs (default: 150)'
    )
    
    args = parser.parse_args()
    
    print("SIAO-CNN-OGRU Training Pipeline - NPPAD Dataset")
    print("=" * 60)
    print(f"Dataset: NuclearPowerPlantAccidentData (NPPAD)")
    print(f"Classes: 14 active research classes")
    print(f"Features: 97 operational parameters")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60 + "\n")
    
    # Pass command-line arguments to the training function
    results = run_complete_pipeline(
        data_dir='data/Operation_csv_data/',
        window_size=100,
        stride=25,
        cnn_embedding_dim=512,
        wks_dim=97,
        rnn_hidden_size=224,
        rnn_num_layers=2,
        active_class_codes=RESEARCH_CLASS_CODES_14,
        n_folds=args.folds,  # Use command-line argument
        bp_epochs=args.epochs,  # Use command-line argument
        siao_pop_size=50,   # Phase 1: increased 30->50 for benchmarking
        siao_max_iter=100,  # Phase 1: was 40
        save_dir='results/'
    )


