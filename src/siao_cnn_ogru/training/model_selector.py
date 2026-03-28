"""Model-selection training pipeline for 14-class NPPAD experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.siao_cnn_ogru.data.class_metadata import RESEARCH_CLASS_CODES_14, build_label_metadata_map
from src.siao_cnn_ogru.data.nppad_loader import NPPADDataPipeline
from src.siao_cnn_ogru.data.window_processor import SlidingWindowProcessor
from src.siao_cnn_ogru.models.model_zoo import create_model, list_available_models
from src.siao_cnn_ogru.reliability import analyze_reliability


def list_supported_models() -> Dict[str, str]:
    """Expose model choices for notebook/config usage."""
    return list_available_models()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_fold_inputs(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, _, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    scaler.fit(X_train_2d)
    X_train_norm = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_val_norm = scaler.transform(X_val_2d).reshape(X_val.shape).astype(np.float32)
    return np.nan_to_num(X_train_norm), np.nan_to_num(X_val_norm)


def _adaptive_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    target_percentile: int = 50,
    min_samples_per_class: int = 6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
            "target_count": target_count,
        }

    min_eligible = min(class_counts[cls_idx] for cls_idx in sampling_strategy.keys())
    k_neighbors = max(1, min(5, min_eligible - 1))

    n_samples, time_steps, n_features = X_train.shape
    X_flat = X_train.reshape(n_samples, time_steps * n_features)
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    X_res, y_res = smote.fit_resample(X_flat, y_train)
    X_res = X_res.reshape(-1, time_steps, n_features).astype(np.float32)
    return X_res, y_res, {
        "applied": True,
        "target_count": target_count,
        "k_neighbors": k_neighbors,
        "classes": sorted(list(sampling_strategy.keys())),
    }


def _train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_class_weights: bool,
    label_smoothing: float,
    patience: int,
    num_classes: int,
) -> Dict[str, Any]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_class_weights:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        weights = len(y_train) / (num_classes * class_counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            seen += int(xb.size(0))

        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_logits = model(X_val_t)
                val_loss = float(criterion(val_logits, y_val_t).item())
            val_preds = val_logits.argmax(dim=1)
            val_acc = float((val_preds == y_val_t).float().mean().item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_t).argmax(dim=1).detach().cpu().numpy()

    return {
        "y_pred": y_pred,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def run_model_training(
    model: str = "cnn",
    data_dir: str = "data/Operation_csv_data/",
    active_class_codes: Optional[Sequence[str]] = None,
    max_timesteps: Optional[int] = 300,
    window_size: int = 100,
    stride: int = 25,
    n_folds: int = 10,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    use_class_weights: bool = True,
    label_smoothing: float = 0.02,
    use_smote: bool = False,
    smote_target_percentile: int = 50,
    smote_min_samples: int = 6,
    patience: int = 15,
    random_seed: int = 42,
    normal_class_code: str = "Normal",
    time_step_hours: float = 1.0,
    save_dir: str = "results/",
    # passthrough params for the proposed model
    siao_pop_size: int = 25,
    siao_max_iter: int = 40,
    bp_epochs: int = 100,
    bp_lr: float = 0.00157,
    fc_dropout: float = 0.164,
) -> Dict[str, Any]:
    """
    Train one selected model with grouped CV.

    Supported model values:
      cnn, lstm, bilstm, ornn, cnn_ornn, siao_cnn_ogru
    """
    model_key = model.strip().lower()
    if model_key == "siao_cnn_ogru":
        from train_pipeline import run_complete_pipeline

        return run_complete_pipeline(
            data_dir=data_dir,
            max_timesteps=max_timesteps,
            window_size=window_size,
            stride=stride,
            active_class_codes=active_class_codes or RESEARCH_CLASS_CODES_14,
            n_folds=n_folds,
            siao_pop_size=siao_pop_size,
            siao_max_iter=siao_max_iter,
            bp_epochs=bp_epochs,
            bp_lr=bp_lr,
            fc_dropout=fc_dropout,
            batch_size=batch_size,
            save_dir=save_dir,
            random_seed=random_seed,
            normal_class_code=normal_class_code,
            time_step_hours=time_step_hours,
        )

    supported = list_available_models()
    if model_key not in supported or model_key == "siao_cnn_ogru":
        raise ValueError(
            f"Unsupported model='{model}'. Available: {', '.join(supported.keys())}"
        )

    _set_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    selected_class_codes = tuple(active_class_codes) if active_class_codes is not None else RESEARCH_CLASS_CODES_14
    pipeline = NPPADDataPipeline(
        data_dir=data_dir,
        max_timesteps=max_timesteps,
        normalization="none",
        handle_missing="interpolate",
        active_class_codes=selected_class_codes,
    )
    X_raw, y_raw = pipeline.run(use_cache=True, cache_dir="data/processed_raw")

    class_codes = [pipeline.label_to_class[idx] for idx in sorted(pipeline.label_to_class)]
    num_classes = len(class_codes)
    class_metadata_map = build_label_metadata_map(class_codes)

    if X_raw.shape[1] == window_size:
        X_windows = X_raw.astype(np.float32)
        y_windows = y_raw.astype(np.int64)
        group_ids = np.arange(len(y_raw), dtype=np.int64)
    else:
        processor = SlidingWindowProcessor(window_size=window_size, stride=stride, padding="zero")
        X_windows, y_windows, group_ids = processor.transform(
            X_raw,
            y_raw,
            return_sample_indices=True,
        )
        X_windows = X_windows.astype(np.float32)

    class_counts = np.bincount(y_windows, minlength=num_classes)
    active_labels = np.where(class_counts > 0)[0]
    min_windows_per_class = int(class_counts[active_labels].min())
    min_groups_per_class = int(
        min(len(np.unique(group_ids[y_windows == cls])) for cls in active_labels)
    )
    effective_folds = min(n_folds, min_windows_per_class, min_groups_per_class)
    if effective_folds < 2:
        raise ValueError("Not enough samples per class for cross-validation.")

    splitter_name = "StratifiedGroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=random_seed,
        )
        fold_iter = splitter.split(X_windows, y_windows, groups=group_ids)
    except Exception:
        splitter_name = "StratifiedKFold (fallback)"
        splitter = StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=random_seed,
        )
        fold_iter = splitter.split(X_windows, y_windows)

    fold_accuracies = []
    fold_macro_f1 = []
    fold_history = []
    oof_y_true = []
    oof_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter, start=1):
        X_train, X_val = X_windows[train_idx], X_windows[val_idx]
        y_train, y_val = y_windows[train_idx], y_windows[val_idx]

        X_train, X_val = _normalize_fold_inputs(X_train, X_val)

        smote_info = {"applied": False, "reason": "disabled"}
        if use_smote:
            X_train, y_train, smote_info = _adaptive_smote(
                X_train,
                y_train,
                num_classes=num_classes,
                target_percentile=smote_target_percentile,
                min_samples_per_class=smote_min_samples,
            )

        fold_model = create_model(
            model_name=model_key,
            input_features=int(X_train.shape[2]),
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        fold_result = _train_one_fold(
            model=fold_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            use_class_weights=use_class_weights,
            label_smoothing=label_smoothing,
            patience=patience,
            num_classes=num_classes,
        )

        y_pred = fold_result["y_pred"]
        fold_acc = accuracy_score(y_val, y_pred)
        fold_f1 = f1_score(y_val, y_pred, average="macro")

        fold_accuracies.append(float(fold_acc))
        fold_macro_f1.append(float(fold_f1))
        oof_y_true.append(y_val.copy())
        oof_y_pred.append(y_pred.copy())
        fold_history.append(
            {
                "fold": fold_idx,
                "accuracy": float(fold_acc),
                "macro_f1": float(fold_f1),
                "best_epoch": int(fold_result["best_epoch"]),
                "best_val_loss": float(fold_result["best_val_loss"]),
                "smote": smote_info,
                "history": fold_result["history"],
            }
        )

    oof_true = np.concatenate(oof_y_true) if oof_y_true else np.array([], dtype=np.int64)
    oof_pred = np.concatenate(oof_y_pred) if oof_y_pred else np.array([], dtype=np.int64)
    reliability_report = analyze_reliability(
        y_true=oof_true,
        y_pred=oof_pred,
        class_codes=class_codes,
        normal_class_code=normal_class_code,
        time_step_hours=time_step_hours,
    )

    avg_acc = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
    std_acc = float(np.std(fold_accuracies)) if fold_accuracies else 0.0
    avg_macro_f1 = float(np.mean(fold_macro_f1)) if fold_macro_f1 else 0.0

    results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"training_summary_{model_key}_14class.json"
    reliability_path = results_dir / f"reliability_summary_{model_key}_14class.json"

    summary_payload = {
        "model": model_key,
        "cv_splitter": splitter_name,
        "effective_folds": int(effective_folds),
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "avg_macro_f1": avg_macro_f1,
        "fold_accuracies": fold_accuracies,
        "fold_macro_f1": fold_macro_f1,
        "class_codes": class_codes,
        "class_metadata": [
            {
                "label": idx,
                "code": class_metadata_map[idx].code,
                "full_name": class_metadata_map[idx].full_name,
                "source": class_metadata_map[idx].source,
            }
            for idx in range(num_classes)
        ],
        "fold_history": fold_history,
    }

    summary_path.write_text(json.dumps(summary_payload, indent=2))
    reliability_path.write_text(json.dumps(reliability_report, indent=2))

    return {
        **summary_payload,
        "oof_y_true": oof_true,
        "oof_y_pred": oof_pred,
        "reliability": reliability_report,
        "summary_path": str(summary_path),
        "reliability_summary_path": str(reliability_path),
    }
