# SN Explosion Fine-tuning Pipeline (Noctua)

Fine-tunes The Well benchmark models on high-resolution supernova explosion simulations
and compares their accuracy at predicting the state 20 Myr after the explosion.

## Dataset

| Property | Value |
|---|---|
| Fields | `density`, `energy` |
| Spatial grid | 64 × 64 × 64 |
| Input | snapshot at t₀ (explosion) |
| Output | snapshot at t₀ + 20 Myr |
| Source format | HDF5 with `"raw"` (input) and `"label"` (output) datasets, shape `(2, 64, 64, 64)` |

## Models

| Model | Config |
|---|---|
| FNO | `configs/experiment/finetune_fno_sn.yaml` |
| TFNO | `configs/experiment/finetune_tfno_sn.yaml` |
| UNet Classic | `configs/experiment/finetune_unet_classic_sn.yaml` |
| UNet ConvNext | `configs/experiment/finetune_unet_convnext_sn.yaml` |

Pretrained weights are downloaded from
[HuggingFace / polymathic-ai](https://huggingface.co/collections/polymathic-ai/the-well-benchmark-models-67e69bd7cd8e60229b5cd43e)
(`supernova_explosion_64` checkpoint). The input/output projection layers are reinitialized
to match `dim_in=2`, `dim_out=2`; all internal (backbone) weights are transferred.

---

## Prerequisites

```bash
# Install from repo root
pip install -e ".[benchmark]"

# Log in to wandb (used by the training script)
wandb login
```

---

## Quick start (single command)

Edit the two placeholders at the top of `noctua/run_pipeline.sh`, then:

```bash
# From the repository root
bash noctua/run_pipeline.sh
```

This runs all four steps sequentially. See below for running steps individually.

---

## Step-by-step

All commands are run from the **repository root** unless otherwise noted.

### Step 1 — Set simulation path

Open `noctua/run_pipeline.sh` and set:

```bash
SIM_DIR="__PLACEHOLDER__/simulations"   # ← path to your raw .h5 simulation files
```

Each file in `SIM_DIR` must be an HDF5 file with:
- `"raw"`:   shape `(2, 64, 64, 64)` — `[density, energy]` at t₀
- `"label"`: shape `(2, 64, 64, 64)` — `[density, energy]` at t₀ + 20 Myr

### Step 2 — Convert to Well format

```bash
python noctua/convert_data.py \
    --input_dir  __PLACEHOLDER__/simulations \
    --output_dir datasets/sn_explosion_hr/data \
    --train_frac 0.8 \
    --val_frac   0.1 \
    --trajs_per_file 4 \
    --seed 42
```

Output layout:

```
datasets/sn_explosion_hr/data/
  train/  sn_0000.h5  sn_0001.h5  ...
  valid/  sn_0000.h5  ...
  test/   sn_0000.h5  ...
```

### Step 3 — Compute normalization statistics

```bash
python noctua/compute_stats.py \
    --base_path datasets/sn_explosion_hr
```

Writes `datasets/sn_explosion_hr/stats.yaml`. Run this only once (or delete the file to recompute).

### Step 4 — Prepare pretrained checkpoints

```bash
python noctua/prepare_checkpoints.py \
    --output_dir checkpoints/ \
    --models fno tfno unet_classic unet_convnext
```

Downloads pretrained weights, transfers backbone, reinitializes IO layers, and saves:

```
checkpoints/
  finetune_fno.pt
  finetune_tfno.pt
  finetune_unet_classic.pt
  finetune_unet_convnext.pt
```

If a pretrained model is not available on HuggingFace the script falls back to random
initialization and continues — you will see a warning.

### Step 5 — Fine-tune

Run from `the_well/benchmark/`:

```bash
cd the_well/benchmark

# FNO
python train.py \
    experiment=finetune_fno_sn \
    server=local \
    optimizer.lr=5e-5 \
    checkpoint_override=../../checkpoints/finetune_fno.pt

# TFNO
python train.py \
    experiment=finetune_tfno_sn \
    server=local \
    optimizer.lr=5e-5 \
    checkpoint_override=../../checkpoints/finetune_tfno.pt

# UNet Classic
python train.py \
    experiment=finetune_unet_classic_sn \
    server=local \
    optimizer.lr=5e-5 \
    checkpoint_override=../../checkpoints/finetune_unet_classic.pt

# UNet ConvNext
python train.py \
    experiment=finetune_unet_convnext_sn \
    server=local \
    optimizer.lr=5e-5 \
    checkpoint_override=../../checkpoints/finetune_unet_convnext.pt
```

**On Noctua HPC**, replace `server=local` with `server=noctua` and first fill in the
absolute paths in `the_well/benchmark/configs/server/noctua.yaml`:

```yaml
data:
  well_base_path: "__PLACEHOLDER__/datasets/"
experiment_dir:  "__PLACEHOLDER__/experiments/"
```

#### Outputs

```
the_well/benchmark/experiments/
  sn_explosion_hr-finetune_sn-FNO-5e-05/
    0/
      extended_config.yaml     ← full config snapshot
      checkpoints/
        recent.pt              ← saved every epoch (overwritten)
        checkpoint_10.pt       ← saved every 10 epochs
      artifacts/               ← metric logs
      viz/                     ← validation plots
```

#### Resuming an interrupted run

Re-run the exact same command **without** `checkpoint_override`. The trainer detects
`checkpoints/recent.pt` and resumes automatically.

#### Learning rate guidance

| Situation | LR |
|---|---|
| Pretrained transfer from same physics (supernova) | `5e-5` |
| Pretrained transfer from different physics | `1e-4` |
| Training from scratch | `1e-3` |

### Step 6 — Evaluate

After training completes, the best checkpoint for each model is saved as `best.pt` inside
its experiment folder. Copy or symlink them into `checkpoints/` then:

```bash
python noctua/evaluate.py \
    --checkpoint_dir checkpoints/ \
    --dataset_base   datasets/sn_explosion_hr
```

Prints a ranked table and writes `checkpoints/results.csv` with per-field:
MSE, RMSE, relative error, and spectral error.

---

## Monitoring with wandb

All runs log to the wandb project **`the_well_extended_again`**.
Filter by group `sn_explosion_hr` to compare models side by side.

Key metrics to watch:
- `valid_sn_explosion_hr/full_MSE_T=all` — overall validation loss
- `valid_sn_explosion_hr/density_MSE_T=all` — density field accuracy
- `valid_sn_explosion_hr/energy_MSE_T=all` — energy field accuracy

---

## File reference

```
noctua/
  convert_data.py          Step 2 — raw HDF5 → Well HDF5
  compute_stats.py         Step 3 — compute stats.yaml
  prepare_checkpoints.py   Step 4 — download & adapt pretrained weights
  run_pipeline.sh          Steps 2–5 in one script
  evaluate.py              Step 6 — test-set comparison
  README.md                This file

the_well/benchmark/configs/
  data/sn_explosion_hr.yaml              Dataset config
  experiment/finetune_fno_sn.yaml        FNO experiment
  experiment/finetune_tfno_sn.yaml       TFNO experiment
  experiment/finetune_unet_classic_sn.yaml
  experiment/finetune_unet_convnext_sn.yaml
  server/noctua.yaml                     Noctua HPC paths
```

---

## Common issues

**`FileNotFoundError: No HDF5 files found`**
Check `--input_dir` in Step 2 or `--base_path` in Step 3.

**`AssertionError: Multiple dataset names found`**
All output `.h5` files must have `f.attrs["dataset_name"] == "sn_explosion_hr"`.
Re-run `convert_data.py` to regenerate.

**`std is abnormally low` in compute_stats**
One field has near-zero variance in the training set. Inspect the physical range of your
density / energy fields.

**CUDA out of memory**
Reduce `batch_size` in `the_well/benchmark/configs/data/sn_explosion_hr.yaml`.
Start with `batch_size: 2` for 3-D 64³ data on a 24 GB GPU.

**HuggingFace download fails**
`prepare_checkpoints.py` falls back to random init automatically. Alternatively,
download manually and pass the local path to `model_cls.from_pretrained(local_dir)`.
