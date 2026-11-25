# Ensemble disagreement vs. softmax confidence for rejecting unseen attack families in IoT intrusion detection

Code, raw results and evaluation scripts for the paper *"Ensemble Disagreement Versus Softmax Confidence for Rejecting Unseen Attack Families in IoT Intrusion Detection: A Leave-One-Group-Out Study on Three Public Datasets"*.

Author: Andi Setiawan. Paper reference: *[to be added after publication]*.

The study withholds one attack family at a time (leave-one-group-out, LOGO), trains tree-based classifiers on the remaining families, and compares ten scores for rejecting the withheld family on Edge-IIoTset, CICIoT2023 and TON_IoT (5 classifiers, 5 seeds, paired Wilcoxon tests). Everything runs on CPU.

## Repository layout

```
src/                     scripts (the experiments are plain .py files with "# %%" cells)
  notebook_d.py            LOGO experiments, Edge-IIoTset and CICIoT2023 (DATASET=edge|cic)
  notebook_e.py            same experiments for TON_IoT (DATASET=ton), also MI over random-forest trees
  notebook_f.py            random-forest-only run, MI over trees (DATASET=edge|cic)
  notebook_g.py            evaluation: rebuilds every table/statistic of the paper and checks 373 numbers
  make_figures.py          Figures 2-6
kaggle_notebooks/        the same scripts as .ipynb, with DATASET fixed, ready to run on Kaggle
results/
  raw/<dataset>/           raw outputs of the experiment runs (CSV, scores_seed0.npz, logs, figures)
  evaluation_inputs/       the 21 CSV files read by notebook_g.py and make_figures.py
  evaluation_outputs/      tables, figures and cek_angka_paper.csv produced by notebook_g.py
figures/                 Figure 1 (draw.io source + PNG) and Figures 2-6
archive/                 earlier versions, NOT used in the paper (see archive/README.md)
```

## Datasets (not included)

The raw datasets are public and are not redistributed here. Obtain them from the original authors; the Kaggle copies used in the experiments are listed for convenience.

| Dataset | Kaggle copy used | File used |
|---|---|---|
| Edge-IIoTset (Ferrag et al., 2022) | `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot` | `Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv` |
| CICIoT2023 (Neto et al., 2023) | `himadri07/ciciot2023` | `CICIOT23/train/train.csv` |
| TON_IoT network data (Moustafa, 2021) | `arnobbhowmik/ton-iot-network-dataset` | `train_test_network.csv` (a curated subset of the full TON_IoT network data) |

Preprocessing is inside the scripts: identifier and high-cardinality text columns, IP addresses and label leaks are removed, duplicates and constant columns are dropped, and port columns are **excluded** from the main experiments (they are re-added in a separate sensitivity run). CICIoT2023 is sampled with a cap of 4,000 flows per attack type and 30,000 benign flows.

## Reproducing the results

### 1. Experiments

On Kaggle: create a notebook from a file in `kaggle_notebooks/`, attach the dataset listed above, and run it.

Locally (set `DATA` to the dataset file, `OUT` to an output folder; `QUICK=1` runs a 1-seed smoke test):

```bash
pip install -r requirements.txt
DATASET=edge DATA=/path/ML-EdgeIIoT-dataset.csv OUT=out_edge python src/notebook_d.py
DATASET=cic  DATA=/path/train.csv               OUT=out_cic  python src/notebook_d.py
DATASET=ton  DATA=/path/train_test_network.csv  OUT=out_ton  python src/notebook_e.py
DATASET=edge DATA=/path/ML-EdgeIIoT-dataset.csv OUT=out_rf_edge python src/notebook_f.py   # random forest only
DATASET=cic  DATA=/path/train.csv               OUT=out_rf_cic  python src/notebook_f.py
```

Seeds are 0-4 at the family level (0-2 for the type-level run on Edge-IIoTset). The full runs take hours on a CPU; observed runtimes are in the logs under `results/raw/`.

### 2. Evaluation (rebuilds the paper's tables and checks them)

```bash
DATA=results/evaluation_inputs OUT=results/evaluation_outputs python src/notebook_g.py
```

The last cell compares 373 numbers quoted in the paper with the recomputed values and writes `cek_angka_paper.csv`.

### 3. Figures

```bash
python src/make_figures.py        # writes figures/fig2 ... fig6
```

Figure 1 is `figures/fig1_logo_protocol.drawio` (open at app.diagrams.net).

## Where each table and figure of the paper comes from

| Paper | File |
|---|---|
| Table 1 (prior work), Table 3 (settings) | descriptive; settings are in `src/notebook_d.py` (`make`, `feature_scores`) |
| Table 2 (datasets) | flow counts printed in `results/raw/*/*.log` |
| Table 4 closed-world F1 | `results/evaluation_outputs/T1_closed_world_f1.csv` |
| Table 5 detection with XGBEns | `T2_deteksi_xgbens.csv`, tests in `T_uji_wilcoxon_xgbens.csv` |
| Table 6 classifier effect | `T3_pengaruh_classifier.csv` |
| Table 7 random forest, MI over trees | `T4_random_forest_mi_pohon.csv`, tests in `T4b_uji_wilcoxon_rf.csv` |
| Table 8 operating points | `T5_operating_points.csv` |
| Table 9 mixed-stream error | `T6_mixed_stream.csv` |
| Table 10 pseudo-unseen selection | `T7_pseudo_ood.csv` |
| Port sensitivity, type-level, mechanism | `T_sensitivitas_port.csv`, `T_tingkat_tipe_edge.csv`, `mekanisme_*.csv` |
| Figures 2-6 | `src/make_figures.py` (inputs: `results/evaluation_inputs`, `results/raw/*/scores_seed0.npz`) |

(The file names `T1`-`T7` follow an earlier numbering of the tables; the mapping above is the current one.)

## Design choices and limitations to keep in mind

- Attack families follow the dataset papers where they define them; several assignments are judgement calls (for example uploading attacks as injection, password attacks as malware). The mapping is in the loaders of `notebook_d.py` and `notebook_e.py`.
- Port columns are excluded from the main results because they act as shortcut features on Edge-IIoTset.
- CICIoT2023 is sampled and TON_IoT is a curated subset; class proportions differ from the original data.
- LightGBM uses `min_child_weight=1`; with the library default the multiclass objective diverged in some runs.
- The statistical tests use (held-out group, seed) pairs that share data, so p-values are indicative.
- Results are for three laboratory datasets and may not transfer to production traffic.

## License

MIT (see `LICENSE`). The datasets keep their own licenses and terms of use.
