# SIAO-CNN-OGRU

Self-Improved Aquila Optimizer enhanced CNN-OGRU pipeline for Nuclear Power Plant Accident Detection.

This repository is the working research codebase for a **14-class modified NPPAD setup**.
It includes training, tuning, reliability analysis, and Colab-ready notebooks.

---

## Current Dataset Configuration

This repo uses a modified subset of NPPAD:

- Original NPPAD definition: **18 classes**
- Active in this project: **14 classes**
- In-house simulated classes: **`Normal`** and **`TT`**
- Current local class folders: `FLB, LLB, LOCA, LOCAC, LR, MD, Normal, RI, RW, SGATR, SGBTR, SLBIC, SLBOC, TT`

Detailed dataset provenance is documented in:

- `data/Readme.md`

---

## Important: Data Handling (No Git LFS)

This workflow does **not** require Git LFS.

You can upload your dataset zip directly in Colab:

- expected zip name: `Operation_csv_data.zip`
- expected extracted path: `data/Operation_csv_data/`

Example Colab commands:

```bash
!mkdir -p data
!rm -rf data/Operation_csv_data
!unzip -q /content/Operation_csv_data.zip -d data/
```

After extraction, confirm folders:

```bash
!ls data/Operation_csv_data
```

---

## Repository Layout

```text
siao-cnn-ogru/
├── train_pipeline.py
├── hyperparameter_tuning.py
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── Operation_csv_data/               # uploaded/extracted dataset (ignored by git)
│   └── Readme.md                         # dataset provenance + class disclosure
├── notebooks/
│   ├── 00_Colab_Training.ipynb
│   ├── 01_SIAO_CNN_OBIGRU_Training.ipynb
│   └── 02_Colab_Training_Reliability_14Class.ipynb
├── src/
│   ├── Readme.md
│   └── siao_cnn_ogru/
│       ├── data/
│       │   ├── class_metadata.py         # 14-class mapping + metadata
│       │   ├── nppad_loader.py
│       │   └── window_processor.py
│       ├── features/
│       ├── models/
│       ├── optimizers/
│       └── visualization/
└── results/
```

---

## 14-Class Metadata

Class mapping and source metadata are centralized in:

- `src/siao_cnn_ogru/data/class_metadata.py`

Key objects:

- `RESEARCH_CLASS_CODES_14`
- `SIMULATED_CLASS_CODES`
- `build_label_maps(...)`
- `build_label_metadata_map(...)`

Data loading uses contiguous labels from active class codes via:

- `NPPADDataPipeline(..., active_class_codes=...)`

---

## Training Pipeline

Main script:

```bash
python train_pipeline.py
```

What it does:

1. Loads and preprocesses sequence data
2. Creates sliding windows
3. Extracts CNN + statistical + WKS features
4. Runs SIAO-initialized OGRU/ORNN training
5. Evaluates with stratified cross-validation
6. Saves plots/results under `results/`

---

## Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

Runs Optuna tuning on the current 14-class active setup and writes output to `results/`.

---

## Colab (Single Notebook Workflow)

Use:

- `notebooks/02_Colab_Training_Reliability_14Class.ipynb`

This notebook includes:

- environment setup
- dataset validation
- model training
- reliability analysis (`λ`, `MTTF`, `R(t)`)
- baseline metric comparison (RMSE, MAE, EVS, R²)
- zipping and downloading `results/`

---

## Installation (Local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If using GPU, install a CUDA-compatible PyTorch build for your environment.
