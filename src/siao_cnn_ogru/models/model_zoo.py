"""Model zoo for side-by-side fault-classification experiments."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .ornn_model import ORNN


class CNNClassifier(nn.Module):
    """Simple 1D CNN classifier over time-series."""

    def __init__(self, input_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        z = self.net(x.transpose(1, 2))
        return self.head(z)


class LSTMClassifier(nn.Module):
    """LSTM / BiLSTM classifier."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class ORNNClassifier(nn.Module):
    """ORNN baseline without SIAO optimization."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "gru",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.ornn = ORNN(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type=cell_type,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.ornn.get_output_size(), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.ornn(x)
        return self.head(out[:, -1, :])


class CNNORNNClassifier(nn.Module):
    """CNN front-end + ORNN sequence model baseline."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        cnn_channels: int = 96,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        self.ornn = ORNN(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type="gru",
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.ornn.get_output_size(), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.ornn(z)
        return self.head(out[:, -1, :])


def create_model(
    model_name: str,
    input_features: int,
    num_classes: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory for baseline models."""
    key = model_name.strip().lower()
    if key == "cnn":
        return CNNClassifier(input_features=input_features, num_classes=num_classes, dropout=dropout)
    if key == "lstm":
        return LSTMClassifier(
            input_features=input_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )
    if key == "bilstm":
        return LSTMClassifier(
            input_features=input_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
    if key == "ornn":
        return ORNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="gru",
            bidirectional=True,
        )
    if key == "cnn_ornn":
        return CNNORNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
    raise ValueError(
        f"Unsupported model='{model_name}'. "
        "Choose from: cnn, lstm, bilstm, ornn, cnn_ornn, siao_cnn_ogru."
    )


def list_available_models() -> Dict[str, str]:
    """Human-readable model registry."""
    return {
        "cnn": "1D-CNN baseline",
        "lstm": "Unidirectional LSTM baseline",
        "bilstm": "Bidirectional LSTM baseline",
        "ornn": "Optimized-RNN architecture (without SIAO pre-optimization)",
        "cnn_ornn": "CNN + ORNN baseline",
        "siao_cnn_ogru": "Proposed model via run_complete_pipeline",
    }

