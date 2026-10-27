# Deep-Sea Classification: Context Reliance & Depth Degradation

Does a deep-sea classification model's reliance on background context predict how badly it degrades at depth?


## Run it in Google Colab (recommended)

1. **`train.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kay-asp/cv-project/blob/main/train.ipynb)\
   builds crops/masks, trains both models (shallow-only classifier and
   the all-depths control), runs predictions, Grad-CAM + CRS, and background interventions (multi-hour run). Writes
   all CSVs to `results/` and checkpoints to `checkpoints/`
   
   
3. **`analysis.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kay-asp/cv-project/blob/main/analysis.ipynb) \
   reads `results/*.csv` and produces the statistics and plots
   (CRS–degradation correlation, baselines, bootstrap CIs, etc.). No GPU required.
   

Click the Colab link for each notebook to open and run in Google Colab. The notebook's first cell clones this public repository and
installs dependencies; `train.ipynb` then downloads the FathomNet metadata it needs (No Google Drive access needed). \
\
For `train.ipynb`:

1. Set the runtime to a GPU: **Runtime → Change runtime type → T4 GPU**.
2. **Runtime → Run all**.

> **Colab will not save results outside of the virtual memory**
> it allows a full run & reproduction from scratch, and produces every training output, but everything a run produces
> (crops, masks, checkpoints, `results/`) is written to the temporary Colab VM and is lost when the
> session ends or disconnects, unless you download it during the session — nothing is written back to this repo or to Drive. The committed
> `results/` already contains our run, so `analysis.ipynb` reproduces the figures without retraining.
> **If you want to do a clean run and keep the new results, clone and run locally** (see below).


## Run it locally

Requires Python 3.10+ and, for training, a CUDA GPU (CPU works but is slow).

```bash
git clone https://github.com/kay-asp/cv-project.git
cd cv-project
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Launch Jupyter from the repo root so the notebooks' relative paths resolve. Then open and run
`train.ipynb`. To run a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute train.ipynb
```

> **Windows note:** `pycocotools` needs Microsoft C++ Build Tools to install. If that's a problem,
> use the Colab path above, which already has it.

None of the data is committed to git — it's downloaded or built on the first run, so the repo
stays small.

- **`data/train.json` is downloaded, not committed.** It's the FathomNet COCO metadata (~280 MB —
  too large for git), so the first cell of section 1 in `train.ipynb` fetches it once into `data/`
  and reuses it on later runs. It's the only metadata file the pipeline needs.
- **`data/depth_cache.csv` ships with the repo.** It's the FathomNet per-image depth lookups
  (~2 MB), committed so a fresh run reuses it instead of crawling the API. It's rebuilt
  automatically if you delete it.
- **Crops and masks are built automatically on the first run.** `train.ipynb` downloads the
  source images from [FathomNet](https://www.fathomnet.org/) and crops them into `crops/` (masks
  are derived from the COCO annotations into `masks/`). This needs an internet connection, is
  resumable (already-built crops are skipped), and can take a while on a cold run. No shared drive
  or external hosting is involved.


## Reproducibility

- **`comparing-the-runs.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kay-asp/cv-project/blob/main/comparing-the-runs.ipynb) \
     checks that seeded training reproduces, by comparing the validation-accuracy histories of two labelled runs (`HISTORY_RUN = "run1"` / `"run2"`).
- The random seed is `config.RANDOM_SEED = 42`, applied via `model.set_seed()` (seeds `random`,
  `numpy`, `torch`, CUDA, and sets `cudnn.deterministic`).
- The train/val/test split is deterministic — a hash of `(seed, image_id)` — so every run and
  every DataLoader sees the same split.
- Dependencies are pinned in `requirements.txt`.
- **Reference environment** (produced the committed `results/`): Google Colab, Tesla T4 GPU,
  `torch 2.11.0+cu130`, CUDA 13.0. Running on this environment reproduces the saved
  validation-accuracy histories exactly (`comparing-the-runs.ipynb` → `np.allclose` is `True`).
  Other GPUs/driver stacks will produce very close but not necessarily bit-identical numbers.

## Project structure

```
├── config.py                 # Hyperparameters, paths, depth thresholds, seed
├── data.py                   # FathomNet loading, cropping, mask building, depth splitting
├── model.py                  # ResNet-50 setup, fine-tuning, set_seed()
├── attribution.py            # Grad-CAM and Context Reliance Score (CRS)
├── evaluation.py             # Accuracy, degradation, baselines
├── analysis.py               # Spearman correlation, scatter plots
├── extensions.py             # Background interventions and swaps
├── train.ipynb               # Main pipeline: build data, train, attribute, save results
├── analysis.ipynb            # Statistics and plots from saved results (CPU)
├── comparing-the-runs.ipynb  # Seeded-reproducibility check across runs
├── data/                     # depth_cache.csv (committed); train.json downloaded on first run (gitignored)
└── results/                  # Saved CSVs and figures from our run
```

Google Colab Full Links (if 'Open in Colab' link doesn't work):
1. **`train.ipynb`** - https://colab.research.google.com/github/kay-asp/cv-project/blob/main/train.ipynb
2. **`analysis.ipynb`** - https://colab.research.google.com/github/kay-asp/cv-project/blob/main/analysis.ipynb
3. **`comparing-the-runs.ipynb`** -  https://colab.research.google.com/github/kay-asp/cv-project/blob/main/comparing-the-runs.ipynb
