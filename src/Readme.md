# Source Layout (`siao-cnn-ogru`)

This `src/` tree contains the 14-class OGRU research implementation.

## Package
- `src/siao_cnn_ogru/`
  - `data/`: NPPAD loading, class metadata, window processing
  - `features/`: statistical feature extraction
  - `models/`: CNN + OGRU/ORNN model components
  - `optimizers/`: SIAO/Aquila optimization modules
  - `visualization/`: training and confusion-matrix plotting helpers

## 14-Class Configuration
The default data pipeline now targets a 14-class research subset through
`RESEARCH_CLASS_CODES_14` in `data/class_metadata.py`.

- Original NPPAD definition: 18 classes
- Active by default in this repository: 14 classes
- In-house simulated classes: `Normal`, `TT`

## Naming
The package path is now `siao_cnn_ogru` (renamed from `siao_cnn_ornn`).
Top-level scripts (`train_pipeline.py`, `hyperparameter_tuning.py`) import from this path.
