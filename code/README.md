# YeaFiLM-CRVAE

Reproducible code for the GOAI 2026 virtual yeast perturbation solution based on a **YeaFiLM condition/compound backbone** and a **Conditional Residual VAE**.

The repository contains code only. It does not redistribute organizer data, test truth, model checkpoints, W&B credentials, or generated prediction files.

## Repository layout

```text
methods/
  _shared/goai_baselines/              train-only data contract utilities
  b10-a2-film-cross-mlp/               YeaFiLM backbone
  b16-a2-conditional-residual-vae/     final residual CVAE
features/compound-structure/           PubChem/RDKit feature builder
evaluation/official-six-module-scorer/ local six-module evaluator
submission/predict.py                  truth-free checkpoint inference
scripts/                               reproducible command templates
docs/                                  architecture and technical explanation
```

## Model sequence

1. Build the audited PubChem/RDKit compound features.
2. Train the YeaFiLM backbone on the official train split and select its checkpoint with the released validation split.
3. Freeze that checkpoint and train the Conditional Residual VAE.
4. For validation or test inference, call only the CVAE conditional-prior mean. The training-only posterior never receives validation or test proteome values.

## Environment

The frozen reference run used Python 3.13.14, PyTorch 2.13.0+cu132, NumPy 2.4.6, pandas 3.0.3, RDKit 2026.3.5, and W&B 0.28.1. See `requirements.txt`.

## Data placement

Obtain organizer data through the official channel and keep it outside Git. The command templates expect:

```text
data/official/WAYB_WAYC_metadata_train_val(1).csv
data/official/WAYB_WAYC_proteome_raw_train_val.csv
data/inference-metadata/WAYB_WAYC_metadata_test(1).csv
data/compound-structure/contract.csv
data/compound-structure/fingerprints.npz
data/compound-structure/manifest.json
```

The `.gitignore` blocks CSV, NPZ, checkpoints, outputs, and experiment directories by default.

## Quick start: final inference

```bash
python -m pip install -r code/requirements.txt
bash scripts/predict_test.sh
```

The complete deployment and evaluation procedure is in `docs/DEPLOYMENT_AND_EVALUATION.md`. The scripts are located at the package root, one level above this `code/` directory.

## Quick start: retraining after obtaining official train/validation data

This reproducibility copy intentionally excludes the official train/validation metadata and proteome. Obtain them through the organizer's official channel and place them under `data/official/` before retraining. The scripts refuse to overwrite an existing output directory:

```bash
bash scripts/retrain_all.sh retraining/run-$(date +%Y%m%d-%H%M%S) cuda:0
```

For separate stages:

```bash
bash scripts/train_e7.sh retraining/e7-run cuda:0
bash scripts/train_crvae.sh retraining/crvae-run retraining/e7-run cuda:0
```

Use `cpu` if CUDA is unavailable. These commands use `--wandb-mode offline`; no credentials or network access are required. The exact arguments, output files, train-only statistics boundary, and how to use a newly trained E7 checkpoint for CR-VAE are documented in `docs/RETRAINING.md`.

For a fast real-data construction check without a full CR-VAE optimization run:

```bash
bash scripts/smoke_crvae_training.sh verification/crvae-smoke cuda:0
```

## Validation

```bash
python -m compileall methods submission features/compound-structure
pytest -q methods/b10-a2-film-cross-mlp
pytest -q methods/b16-a2-conditional-residual-vae
PYTHONPATH=evaluation/official-six-module-scorer/src pytest -q evaluation/official-six-module-scorer/tests
pytest -q features/compound-structure
```

From the package root, `bash scripts/verify_bundle.sh quick` runs the checksum gate and the bundled tests. Add `inference` to run the full truth-free inference as well.

The scorer is a local implementation of the current handbook contract used during development. The organizer remains the authoritative source for final scoring and submission requirements.
