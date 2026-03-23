# SIAO-CNN-ORNN

Self-Improved Aquila Optimizer enhanced CNN-ORNN for Nuclear Power Plant Accident Detection.

A hybrid deep learning framework that classifies 18 types of nuclear power plant accidents from time-series operational data using the NPPAD (Nuclear Power Plant Accident Data) dataset.

---

## Results

**10-Fold Stratified Cross-Validation**

| Metric | Value |
|--------|-------|
| Average Accuracy | 91.91% (+/- 2.61%) |
| Average Macro-F1 | 91.57% |
| Best Fold | 97.58% |
| Worst Fold | 88.62% |

Per-fold breakdown:

| Fold | Accuracy | Macro-F1 |
|:----:|:--------:|:--------:|
| 1 | 91.13% | 90.36% |
| 2 | 89.52% | 88.60% |
| 3 | 93.55% | 92.92% |
| 4 | 91.94% | 91.62% |
| 5 | 97.58% | 97.37% |
| 6 | 91.94% | 90.96% |
| 7 | 88.62% | 88.80% |
| 8 | 91.87% | 91.74% |
| 9 | 93.50% | 93.24% |
| 10 | 89.43% | 90.04% |

---

## Architecture

```
Raw CSV Data (1,237 samples x 100 timesteps x 96 sensors)
    |
    v
Sliding Windows (window_size=100, stride=25)
    |
    +---> CNN Feature Extractor (pre-trained, 512-dim)
    |
    +---> Statistical Features (mean, median, std, variance, entropy per sensor = 485-dim)
    |
    +---> WKS Features (Weighted Kurtosis-Skewness, Aquila-optimized = 97-dim)
    |
    v
Feature Concatenation (1,094-dim)
    |
    v
SIAO Weight Initialization (population=30, iterations=100)
    |
    v
Bidirectional GRU-ORNN (2 layers, 224 hidden units, 2.67M parameters)
    |
    v
Backpropagation Fine-tuning (Adam, lr=0.00157, AMP, early stopping)
    |
    v
Softmax Classification (13 active classes)
```

### Training Pipeline

1. **Data Loading**: CSV files loaded, padded/truncated to 100 timesteps, z-score normalized
2. **CNN Pre-training**: 30-epoch supervised training with classification head before feature extraction
3. **Feature Extraction**: CNN embeddings + statistical features + WKS features concatenated
4. **SIAO Optimization**: Self-Improved Aquila Optimizer searches for optimal ORNN weight initialization
5. **Backprop Fine-tuning**: Adam optimizer with mixed precision (AMP), StepLR scheduler, early stopping
6. **Evaluation**: Per-fold accuracy and macro-F1, confusion matrix generation

### Class Imbalance Handling

- Adaptive SMOTE: minority classes upsampled to 50th percentile count
- Class-weighted CrossEntropyLoss: inverse-frequency weighting
- Label smoothing: 0.05

---

## Dataset: NPPAD

| Property | Value |
|----------|-------|
| Total Samples | 1,237 |
| Timesteps | 100 (padded/truncated) |
| Features | 96 operational parameters |
| Classes | 18 defined, 13 active in dataset |
| Format | CSV files organized by accident type |

### Accident Classes

| ID | Code | Full Name |
|:--:|:-----|:----------|
| 0 | Normal | Normal Operation |
| 1 | ATWS | Anticipated Transient Without Scram |
| 2 | FLB | Feedwater Line Break |
| 3 | LACP | Loss of AC Power |
| 4 | LLB | Letdown Line Break |
| 5 | LOCA | Loss of Coolant Accident (Hot Leg) |
| 6 | LOCAC | Loss of Coolant Accident (Cold Leg) |
| 7 | LOF | Loss of Flow (Locked Rotor) |
| 8 | LR | Load Rejection |
| 9 | MD | Moderator Dilution |
| 10 | RI | Rod Insertion |
| 11 | RW | Rod Withdrawal |
| 12 | SGATR | Steam Generator A Tube Rupture |
| 13 | SGBTR | Steam Generator B Tube Rupture |
| 14 | SLBIC | Steam Line Break Inside Containment |
| 15 | SLBOC | Steam Line Break Outside Containment |
| 16 | SP | Spark Presence for Hydrogen Burn |
| 17 | TT | Turbine Trip |

The dataset has class imbalance. Some classes have fewer than 10 samples while others have over 100.

---

## Project Structure

```
SIAO-CNN-ORNN/
├── train_pipeline.py                   # Main training script
├── hyperparameter_tuning.py            # Optuna-based hyperparameter search
├── pyproject.toml                      # Project metadata and dependencies
├── requirements.txt                    # Pinned dependencies
├── data/
│   └── Operation_csv_data/             # NPPAD CSV files (18 class folders)
├── src/
│   └── siao_cnn_ornn/
│       ├── data/
│       │   ├── nppad_loader.py         # Data loading, normalization, caching
│       │   └── window_processor.py     # Sliding window extraction
│       ├── features/
│       │   └── feature_extractor.py    # Statistical feature computation
│       ├── models/
│       │   ├── cnn_model.py            # CNN feature extractor (Conv2D)
│       │   ├── ornn_model.py           # Optimized RNN (GRU cells + SIAO trainer)
│       │   ├── classifier.py           # Classification head
│       │   └── model_enhancement.py    # Enhanced ORNN with attention
│       ├── optimizers/
│       │   ├── siao_optimizer.py       # Self-Improved Aquila Optimizer
│       │   └── aquila_optimizer.py     # Aquila Optimizer + WKS computation
│       ├── inference/
│       │   ├── realtime_inference.py   # Real-time prediction
│       │   └── reliability.py          # Reliability analysis
│       └── visualization/
│           └── visualizer.py           # Plotting utilities
├── notebooks/
│   ├── 00_Colab_Training.ipynb         # Google Colab GPU training
│   └── 01_SIAO_CNN_OBIGRU_Training.ipynb  # Comprehensive local training workflow
└── results/
    ├── plots/                          # Training curves, confusion matrices
    └── optuna_best_params.txt          # Best hyperparameters from tuning
```

---

## Hyperparameters

Default values (Optuna-tuned where noted):

| Parameter | Value | Source |
|-----------|-------|--------|
| CNN embedding dim | 512 | Optuna |
| RNN hidden size | 224 | Optuna |
| RNN layers | 2 | Optuna |
| RNN cell type | Bidirectional GRU | Fixed |
| Learning rate | 0.00157 | Optuna |
| FC dropout | 0.164 | Optuna |
| Weight decay | 1.97e-05 | Optuna |
| Batch size | 163 | Fixed |
| BP epochs | 150 | Fixed (early stopping active) |
| Early stopping patience | 20 | Fixed |
| SIAO population | 30 | Fixed |
| SIAO iterations | 100 | Fixed |
| WKS population | 15 | Fixed |
| WKS iterations | 30 | Fixed |
| Label smoothing | 0.05 | Fixed |
| Window size | 100 | Fixed |
| Stride | 25 | Fixed |
| K-folds | 10 | Fixed |

---

## Installation

### Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)

### Local Setup

```bash
git clone https://github.com/sisoodiya/SIAO-CNN-ORNN.git
cd SIAO-CNN-ORNN

python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Google Colab

Open `notebooks/00_Colab_Training.ipynb` in Google Colab. It handles cloning, dependency installation, and training with GPU acceleration.

---

## Usage

### Training

```bash
python train_pipeline.py
```

Runs the full 10-fold cross-validation pipeline with default hyperparameters. Results are saved to `results/`.

### Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

Runs 20 Optuna trials using TPE sampling to optimize RNN hidden size, CNN embedding dim, learning rate, dropout, weight decay, SIAO population/iterations, and number of RNN layers. Results are saved to `results/optuna_best_params.txt`.

---

## Optimizers

### Self-Improved Aquila Optimizer (SIAO)

Finds optimal initial weights for the ORNN before backpropagation fine-tuning. Uses four search strategies based on iteration progress:

- Expanded exploration (Levy flight + population mean)
- Narrowed exploration (Levy flight around best)
- Expanded exploitation (spiral descent)
- Narrowed exploitation (random walk near best)

Includes chaotic map initialization, batch evaluation for GPU efficiency, and convergence-based early stopping.

### Aquila Optimizer (WKS)

Optimizes a single scalar weight `omega` for the Weighted Kurtosis-Skewness feature:

```
WKS = omega * Kurtosis + Skewness
```

Maximizes Fisher Discriminant Ratio (inter-class variance / intra-class variance) for class separability.

---

## Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` or `siao_pop_size` in the training config.

**Slow Training on CPU**: Verify GPU detection with `python -c "import torch; print(torch.cuda.is_available())"`. If False, reinstall PyTorch with CUDA support.

**NaN in Features**: The feature extractor handles NaN/Inf values automatically. If warnings appear, they indicate the source data has missing values in specific sensor channels.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

