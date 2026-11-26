<h1 align="center">PepNet</h1>

<p align="center">
  Predicting the membrane permeability of cyclic peptides with a descriptor-based baseline,
  a custom CNN trained on 2D molecule depictions, and a fine-tuned Vision Transformer.
</p>

<p align="center">
    <a href="LICENSE">
        <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

This repository was built for the **AI4Chemistry** course project by Ziad El Malki, Colleen Lonfat,
and Lennard Reihs. It explores three complementary approaches to predicting cyclic peptide
permeability from the [CycPeptMPDB](http://cycpeptmpdb.com/) database:

1. **Baseline** — gradient boosting / linear models trained on RDKit molecular descriptors
   (TPSA, molecular weight, H-bond donors/acceptors, functional group counts, ...).
2. **PepNet** — a convolutional neural network trained directly on 2D RDKit depictions of
   the peptides, learning permeability from the rendered structure.
3. **ViT** — a fine-tuned [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)
   Vision Transformer, trained on the same molecule images as PepNet.

## Repository structure

```
PepNet/
├── scripts/                 Numbered pipeline scripts, meant to be run in order
├── notebooks/                Exploratory notebooks (baseline tests, ViT tuning)
├── data/                     Generated datasets (empty in git, populated by scripts 01-02)
├── results/                  Metrics, predictions, and plots from our training runs
│   └── plots/
├── environment-linux.yml     Conda environment (Linux)
├── environment-macos.yml     Conda environment (macOS)
└── CITATION.cff
```

## Getting started

### 1. Set up the environment

```shell
conda env create -f environment-linux.yml   # or environment-macos.yml on macOS
conda activate pepnet
```

### 2. Download and clean the data

```shell
cd scripts
python 01_CycPept_Download.py
python 02_CycPept_Cleaning.py
```

This downloads `CycPeptMPDB_Peptide_All.csv` and writes a cleaned version to `data/`.

### 3. Baseline models (gradient boosting, linear models)

```shell
python 07_Baseline.py
```

Trains and grid-searches linear/Ridge/Lasso/Random Forest/Gradient Boosting regressors on
molecular descriptors, and reports MSE/MAE/R² on a held-out test set.

### 4. PepNet (CNN on molecule images)

```shell
python 03_PepNet_DataGenerator.py   # renders SMILES to images, stores them in an HDF5 file
python 04_Data_Augmentor.py         # augments the image dataset
python 05_Train_PepNet.py           # trains the CNN and saves metrics/predictions to results/
```

### 5. ViT fine-tuning

```shell
python 03_PepNet_DataGenerator.py   # if not already run
python 04_Data_Augmentor.py         # if not already run
python 06_ViT_Tuning.py
```

or run interactively with [`notebooks/02_ViT_Tuning.ipynb`](notebooks/02_ViT_Tuning.ipynb).

## Results

Selected metrics from our training runs (see [`results/`](results/) for the full predictions
and plots):

| Model  | Test MSE | Notes |
|--------|----------|-------|
| PepNet (CNN) | 0.540 | 76.2% of predictions within 15% of the true value |
| ViT (fine-tuned) | 0.515 | |

Training curves and prediction scatter plots are available in [`results/plots/`](results/plots/).

## License

This project is licensed under the [MIT License](LICENSE).
