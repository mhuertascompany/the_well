# Fine-tuning The Well Models on Supernova Explosion Data

## Setup summary

| Property | Value |
|---|---|
| Fields | `density`, `energy` (both 3D scalar, rank-0) |
| Spatial grid | 64 × 64 × 64 |
| Time steps per trajectory | 2 (t₀ at explosion, t₀ + 20 Myr) |
| Task | Direct mapping: snapshot at t₀ → snapshot at t₀ + 20 Myr |
| Model input channels | `n_steps_input × n_fields = 1 × 2 = 2` |
| Model output channels | `n_fields = 2` |

---

## Prerequisites

```bash
# Install the Well with benchmark extras
pip install the_well[benchmark]

# Verify
python -c "from the_well.data import WellDataset; from the_well.benchmark.models import FNO"
```

You also need a [wandb](https://wandb.ai) account — the training script logs to it. Login once:

```bash
wandb login
```

---

## Step 1 — Convert your data to the Well HDF5 format

Each `.h5` file holds one or more trajectories. The converter below writes `N_traj` simulation pairs
(input at t₀, output at t₀ + 20 Myr) into a single file. Call it multiple times to spread
trajectories across files (recommended: 4–8 trajectories per file).

Create `scripts/convert_sn_data.py`:

```python
"""Convert SN explosion simulation pairs to Well HDF5 format."""

import h5py
import numpy as np


def write_sn_hdf5(
    filename: str,
    density: np.ndarray,
    energy: np.ndarray,
    sim_params: dict = None,
):
    """
    Args:
        filename:   Output path, e.g. 'train/sn_000.h5'
        density:    shape (N_traj, 2, 64, 64, 64), float32
                    axis 1: [t0, t0+20Myr]
        energy:     shape (N_traj, 2, 64, 64, 64), float32
        sim_params: optional dict of per-trajectory scalar parameters,
                    e.g. {"explosion_energy": array(N_traj,)}
                    These are stored as constant scalars (not used as model input
                    unless you add them to the config, but useful for bookkeeping).
    """
    N, T, Nx, Ny, Nz = density.shape
    assert T == 2, "Expected exactly 2 time steps: t0 and t0+20Myr"
    assert density.shape == energy.shape

    x = np.linspace(0, 1, Nx, dtype=np.float32)
    y = np.linspace(0, 1, Ny, dtype=np.float32)
    z = np.linspace(0, 1, Nz, dtype=np.float32)
    t = np.array([0.0, 20.0], dtype=np.float32)   # Myr
    open_mask = np.zeros(Nx, dtype=bool)           # OPEN BC — no periodic walls

    with h5py.File(filename, "w") as f:
        # Top-level attributes
        f.attrs["dataset_name"] = "sn_explosion_hr"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 3
        f.attrs["n_trajectories"] = N
        f.attrs["simulation_parameters"] = list(sim_params.keys()) if sim_params else []

        # ── Boundary conditions ──────────────────────────────────────────────
        bc = f.create_group("boundary_conditions")
        for ax, mask in zip(["x", "y", "z"], [open_mask] * 3):
            sg = bc.create_group(f"{ax}_open")
            sg.attrs["associated_dims"] = [ax]
            sg.attrs["associated_fields"] = []
            sg.attrs["bc_type"] = "OPEN"
            sg.attrs["sample_varying"] = False
            sg.attrs["time_varying"] = False
            sg.create_dataset("mask", data=mask)

        # ── Dimensions ───────────────────────────────────────────────────────
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y", "z"]
        dims.create_dataset("time", data=t)
        dims["time"].attrs["sample_varying"] = False
        for ax, arr in zip(["x", "y", "z"], [x, y, z]):
            dims.create_dataset(ax, data=arr)
            dims[ax].attrs["sample_varying"] = False

        # ── Scalars (optional per-trajectory simulation parameters) ──────────
        sc = f.create_group("scalars")
        sc.attrs["field_names"] = list(sim_params.keys()) if sim_params else []
        for name, vals in (sim_params or {}).items():
            ds = sc.create_dataset(name, data=np.asarray(vals, dtype=np.float32))
            ds.attrs["time_varying"] = False
            ds.attrs["sample_varying"] = True

        # ── t0_fields: density and energy (rank-0 scalar fields) ─────────────
        # Shape stored in HDF5: (N_traj, T, 64, 64, 64)
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["density", "energy"]
        for name, arr in [("density", density), ("energy", energy)]:
            ds = t0.create_dataset(name, data=arr.astype(np.float32))
            ds.attrs["dim_varying"] = [True, True, True]   # varies in x, y, z
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True

        # ── Required empty groups (no vector or tensor fields) ────────────────
        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = []
        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Replace these with your actual data loading
    # density, energy: (N_total, 2, 64, 64, 64) arrays from your simulations
    N_total = 100   # total number of simulation pairs
    density = np.random.rand(N_total, 2, 64, 64, 64).astype(np.float32)
    energy  = np.random.rand(N_total, 2, 64, 64, 64).astype(np.float32)

    # Split: 80 train / 10 valid / 10 test
    splits = {
        "train": (0, 80),
        "valid": (80, 90),
        "test":  (90, 100),
    }
    trajs_per_file = 5   # number of trajectories per .h5 file

    base = "datasets/sn_explosion_hr/data"
    for split, (start, end) in splits.items():
        os.makedirs(f"{base}/{split}", exist_ok=True)
        chunk_starts = range(start, end, trajs_per_file)
        for file_idx, cs in enumerate(chunk_starts):
            ce = min(cs + trajs_per_file, end)
            write_sn_hdf5(
                filename=f"{base}/{split}/sn_{file_idx:03d}.h5",
                density=density[cs:ce],
                energy=energy[cs:ce],
            )
            print(f"{split}: wrote trajectories {cs}–{ce-1} → sn_{file_idx:03d}.h5")
```

Run it:

```bash
python scripts/convert_sn_data.py
```

---

## Step 2 — Final directory layout

After conversion your dataset directory must look exactly like this:

```
datasets/
  sn_explosion_hr/
    data/
      train/
        sn_000.h5
        sn_001.h5
        ...
      valid/
        sn_000.h5
        ...
      test/
        sn_000.h5
        ...
    stats.yaml          ← computed in Step 3 (does not exist yet)
```

The Well reads the `data/<split>/` folders automatically; `stats.yaml` must sit at the dataset root.

---

## Step 3 — Compute normalization statistics

The existing `compute_statistics.py` script reads all training `.h5` files and writes per-field
mean, std, rms, mean_delta, std_delta, rms_delta to `stats.yaml`.

```bash
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from scripts.compute_statistics import compute_statistics

compute_statistics(
    train_path="datasets/sn_explosion_hr/data/train",
    stats_path="datasets/sn_explosion_hr/stats.yaml",
)
print("Done — stats.yaml written.")
EOF
```

Inspect the result to sanity-check that std values are not near zero:

```bash
python -c "
import yaml
with open('datasets/sn_explosion_hr/stats.yaml') as f:
    s = yaml.safe_load(f)
for field in ['density', 'energy']:
    print(f'{field}: mean={s[\"mean\"][field]:.3e}  std={s[\"std\"][field]:.3e}')
"
```

---

## Step 4 — Dataset config

Create `the_well/benchmark/configs/data/sn_explosion_hr.yaml`:

```yaml
_target_: the_well.data.WellDataModule
well_base_path: ../../datasets/          # relative to the_well/benchmark/
well_dataset_name: sn_explosion_hr
batch_size: 4
use_normalization: True
normalization_type:
  _partial_: true
  _target_: the_well.data.normalization.ZScoreNormalization
n_steps_input: 1    # one snapshot in  (t0)
n_steps_output: 1   # one snapshot out (t0 + 20 Myr)
min_dt_stride: 1
max_dt_stride: 1
```

> **`well_base_path`** is resolved relative to where `train.py` is run
> (`the_well/benchmark/`), so `../../datasets/` points to the repo-root `datasets/` folder.
> Use an absolute path if you store data elsewhere.

---

## Step 5 — Prepare pretrained checkpoints

The Well provides pretrained weights on HuggingFace under
`polymathic-ai/<ModelName>-supernova_explosion_64`. The pretrained supernova models have
`dim_in=24` (4 input steps × 6 fields) and `dim_out=6`, while your data needs `dim_in=2`
and `dim_out=2`. The input and output projection layers must be reinitialized; all internal
layers (the learned physics representation) transfer directly.

Create `scripts/prepare_finetune_checkpoints.py`:

```python
"""
Download pretrained supernova_explosion_64 weights and prepare fine-tuning
checkpoints for models compatible with dim_in=2, dim_out=2, 64^3 3-D grid.

Reinitialized layers (random init, trained from scratch):
  FNO / TFNO / ReFNO : model.lifting  (in_channels → hidden_channels)
                        model.projection (hidden_channels → out_channels)
  UNetClassic        : encoder1       (dim_in → init_features)
                        conv           (init_features → dim_out)
  UNetConvNext       : encoder.0      (dim_in → init_features)
                        decoder_final  (init_features → dim_out)

All other weights are copied from the pretrained checkpoint.
"""

import torch
from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext

DIM_IN  = 2
DIM_OUT = 2
N_DIMS  = 3
RES     = (64, 64, 64)


def transfer_weights(pretrained, new_model):
    """Copy weights where shapes match; skip mismatched IO layers."""
    pretrained_sd = pretrained.state_dict()
    new_sd        = new_model.state_dict()

    transferred = {
        k: v for k, v in pretrained_sd.items()
        if k in new_sd and v.shape == new_sd[k].shape
    }
    skipped = [k for k in pretrained_sd if k not in transferred]

    new_sd.update(transferred)
    new_model.load_state_dict(new_sd)

    print(f"  Transferred : {len(transferred)} tensors")
    print(f"  Reinitialized: {skipped}")
    return new_model


def make_checkpoint(model, path):
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dit": {},            # optimizer reinits from scratch
        "validation_loss": float("inf"),
        "best_validation_loss": float("inf"),
    }, path)
    print(f"  Saved → {path}")


# ── FNO ──────────────────────────────────────────────────────────────────────
print("FNO")
pretrained = FNO.from_pretrained("polymathic-ai/FNO-supernova_explosion_64")
model = FNO(
    dim_in=DIM_IN, dim_out=DIM_OUT,
    n_spatial_dims=N_DIMS, spatial_resolution=RES,
    modes1=16, modes2=16, modes3=16,
    hidden_channels=128,
)
model = transfer_weights(pretrained, model)
make_checkpoint(model, "checkpoints/finetune_fno.pt")

# ── TFNO ─────────────────────────────────────────────────────────────────────
print("TFNO")
pretrained = TFNO.from_pretrained("polymathic-ai/TFNO-supernova_explosion_64")
model = TFNO(
    dim_in=DIM_IN, dim_out=DIM_OUT,
    n_spatial_dims=N_DIMS, spatial_resolution=RES,
    modes1=16, modes2=16, modes3=16,
    hidden_channels=128,
)
model = transfer_weights(pretrained, model)
make_checkpoint(model, "checkpoints/finetune_tfno.pt")

# ── UNetClassic ───────────────────────────────────────────────────────────────
print("UNetClassic")
pretrained = UNetClassic.from_pretrained("polymathic-ai/UNetClassic-supernova_explosion_64")
model = UNetClassic(
    dim_in=DIM_IN, dim_out=DIM_OUT,
    n_spatial_dims=N_DIMS, spatial_resolution=RES,
    init_features=48,
)
model = transfer_weights(pretrained, model)
make_checkpoint(model, "checkpoints/finetune_unet_classic.pt")

# ── UNetConvNext ──────────────────────────────────────────────────────────────
print("UNetConvNext")
pretrained = UNetConvNext.from_pretrained("polymathic-ai/UNetConvNext-supernova_explosion_64")
model = UNetConvNext(
    dim_in=DIM_IN, dim_out=DIM_OUT,
    n_spatial_dims=N_DIMS, spatial_resolution=RES,
    init_features=42, blocks_per_stage=2,
)
model = transfer_weights(pretrained, model)
make_checkpoint(model, "checkpoints/finetune_unet_convnext.pt")
```

```bash
mkdir -p checkpoints
python scripts/prepare_finetune_checkpoints.py
```

> **If a pretrained model for `supernova_explosion_64` is not available** on HuggingFace,
> use `active_matter` or `turbulent_radiative_layer_3D` as the source checkpoint.
> Any 3-D physics dataset is a better starting point than random initialization.
> If no 3-D pretrained model exists for a given architecture, skip the transfer step and
> train from scratch — the rest of the pipeline is identical.

---

## Step 6 — Experiment configs

Create one config file per model. All configs inherit the same data and trainer settings;
only the model and checkpoint path differ.

### `the_well/benchmark/configs/experiment/finetune_fno_sn.yaml`

```yaml
# @package _global_
defaults:
  - /data: sn_explosion_hr
  - /model: fno
  - /lr_scheduler: cosine_with_warmup

trainer:
  loss_fn:
    _target_: the_well.benchmark.metrics.MSE
  formatter: channels_first_default
  epochs: 100
  val_frequency: 1
  rollout_val_frequency: 0    # disable — T=2 means rollout == one-step, no added value
  checkpoint_frequency: 10

name: finetune_sn
```

### `the_well/benchmark/configs/experiment/finetune_tfno_sn.yaml`

```yaml
# @package _global_
defaults:
  - /data: sn_explosion_hr
  - /model: tfno
  - /lr_scheduler: cosine_with_warmup

trainer:
  loss_fn:
    _target_: the_well.benchmark.metrics.MSE
  formatter: channels_first_default
  epochs: 100
  val_frequency: 1
  rollout_val_frequency: 0
  checkpoint_frequency: 10

name: finetune_sn
```

### `the_well/benchmark/configs/experiment/finetune_unet_classic_sn.yaml`

```yaml
# @package _global_
defaults:
  - /data: sn_explosion_hr
  - /model: unet_classic
  - /lr_scheduler: cosine_with_warmup

trainer:
  loss_fn:
    _target_: the_well.benchmark.metrics.MSE
  formatter: channels_first_default
  epochs: 100
  val_frequency: 1
  rollout_val_frequency: 0
  checkpoint_frequency: 10

name: finetune_sn
```

### `the_well/benchmark/configs/experiment/finetune_unet_convnext_sn.yaml`

```yaml
# @package _global_
defaults:
  - /data: sn_explosion_hr
  - /model: unet_convnext
  - /lr_scheduler: cosine_with_warmup

trainer:
  loss_fn:
    _target_: the_well.benchmark.metrics.MSE
  formatter: channels_first_default
  epochs: 100
  val_frequency: 1
  rollout_val_frequency: 0
  checkpoint_frequency: 10

name: finetune_sn
```

---

## Step 7 — Run fine-tuning

All commands are run from `the_well/benchmark/`. The `checkpoint_override` argument
loads the pretrained weights prepared in Step 5 without resuming optimizer state or epoch
counter — it is a clean fine-tune start.

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

### Learning rate guidance

| Scenario | Recommended LR |
|---|---|
| Fine-tuning from same-physics pretrained weights | 5e-5 – 1e-4 |
| Fine-tuning from different-physics pretrained weights | 1e-4 – 5e-4 |
| Training from scratch | 1e-3 (default) |

Use the same LR for all models when comparing so that LR is not a confound.

### Resuming an interrupted run

The trainer auto-saves `checkpoints/recent.pt` every epoch. To resume, just re-run the
same command without `checkpoint_override` — `configure_experiment` will detect
`recent.pt` and resume automatically.

### Outputs

Each run saves to:

```
the_well/benchmark/experiments/
  sn_explosion_hr-finetune_sn-FNO-5e-05/
    0/                         ← run index, increments on re-run
      extended_config.yaml     ← full resolved config snapshot
      checkpoints/
        recent.pt              ← latest epoch (overwritten each epoch)
        checkpoint_10.pt       ← periodic saves
        best.pt                ← best validation loss so far
      artifacts/               ← metric logs
      viz/                     ← validation plots
```

---

## Step 8 — Compare results

### Via wandb

All runs log to the wandb project `the_well_extended_again` (set in `config.yaml`).
Go to your wandb workspace, filter by group `sn_explosion_hr`, and compare
`valid_sn_explosion_hr/full_MSE_T=all` across models.

### Manually load best checkpoint and evaluate

```python
import torch
from the_well.data import WellDataset
from the_well.benchmark.models import FNO
from the_well.benchmark.metrics import MSE, RMSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_ds = WellDataset(
    well_base_path="datasets/",
    well_dataset_name="sn_explosion_hr",
    well_split_name="test",
    use_normalization=True,
    normalization_type=...,   # ZScoreNormalization partial
    n_steps_input=1,
    n_steps_output=1,
)

# Load best FNO checkpoint
ckpt = torch.load("the_well/benchmark/experiments/.../checkpoints/best.pt",
                  weights_only=False)
model = FNO(dim_in=2, dim_out=2, n_spatial_dims=3, spatial_resolution=(64,64,64),
            modes1=16, modes2=16, modes3=16, hidden_channels=128).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run one-step evaluation
from torch.utils.data import DataLoader
from einops import rearrange

loader = DataLoader(test_ds, batch_size=2, shuffle=False)
mse_fn = MSE()
total_mse = 0.0

with torch.no_grad():
    for batch in loader:
        x = rearrange(batch["input_fields"], "b t ... c -> b (t c) ...").to(device)
        y_ref = batch["output_fields"].to(device)           # (B, 1, 64, 64, 64, 2)
        y_pred = rearrange(model(x), "b c ... -> b 1 ... c")
        total_mse += mse_fn(y_pred, y_ref, test_ds.metadata).mean().item()

print(f"Test MSE: {total_mse / len(loader):.4e}")
```

### Key metrics to compare

| Metric | What it tells you |
|---|---|
| MSE per field | Overall accuracy for density and energy separately |
| RMSE per field | Same in original units (easier to interpret physically) |
| Spectral error | Whether the model captures fine spatial structure |
| val loss curve | Whether fine-tuning converged or overfit |

The spectral error and per-field breakdown are logged automatically by the trainer's
`validation_metric_suite`. Check the wandb `valid_*` keys for each.

---

## Common issues

**`AssertionError: No HDF5 files found`**
The `data_path` resolves incorrectly. Check `well_base_path` in the dataset config —
use an absolute path to be safe.

**`AssertionError: Multiple dataset names found`**
The `dataset_name` attribute in one of your `.h5` files doesn't match the others.
All files in a split must have `f.attrs["dataset_name"] == "sn_explosion_hr"`.

**`std is abnormally low` during `compute_statistics`**
One of your fields has near-zero variance in the training set (e.g. energy is constant
across all your simulations). Check the physical range of that field.

**`KeyError: optimizer_state_dit`** when loading checkpoint
The checkpoint was not saved with the Well trainer format. Re-run
`prepare_finetune_checkpoints.py` to produce a correctly structured file.

**CUDA out of memory**
Reduce `batch_size` in `sn_explosion_hr.yaml`. For 3-D 64³ data, `batch_size=2` is a
safe starting point on a 24 GB GPU. FNO and TFNO are more memory-efficient than UNet
at this resolution.
