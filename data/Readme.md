# Modified NPPAD Dataset for `siao-cnn-ogru`

## Overview
This `data/` directory is based on the original **NPPAD (Nuclear Power Plant Accident Data)** dataset,
but the version used in this research workspace is a **modified 14-class subset**.

- Original NPPAD release: **18 classes**
- This project: **14 classes**
- `Normal` and `TT` in this workspace are **simulated in-house** for this research pipeline.

## Important Disclosure
The data in this repository is **not an unchanged mirror** of the original NPPAD repository.
It is a project-specific research dataset prepared for model training under data-scarcity constraints.

## Classes Used in This Workspace
Folder source: `data/Operation_csv_data/`

| Class | CSV Files |
|---|---:|
| FLB | 100 |
| LLB | 101 |
| LOCA | 100 |
| LOCAC | 100 |
| LR | 99 |
| MD | 100 |
| Normal | 30 |
| RI | 100 |
| RW | 100 |
| SGATR | 100 |
| SGBTR | 110 |
| SLBIC | 101 |
| SLBOC | 100 |
| TT | 26 |

**Total classes:** 14  
**Total samples:** 1267

## Original Dataset Attribution
Original NPPAD dataset and materials are from:
- GitHub: https://github.com/thu-inet/NuclearPowerPlantAccidentData

Please cite and acknowledge the original NPPAD authors when using the base dataset,
and separately disclose this workspace’s class/subset and simulation modifications.

## Scope Note
This README documents dataset provenance and class composition for `siao-cnn-ogru` only.
Model architecture and training pipeline details are maintained at the repository root.
