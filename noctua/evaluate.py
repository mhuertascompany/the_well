"""
Evaluate and compare fine-tuned models on the sn_explosion_hr test set.

For each model checkpoint found in CHECKPOINT_DIR, computes:
  - MSE and RMSE per field (density, energy)
  - Mean relative error per field
  - Spectral error (power spectrum comparison)

Prints a ranked comparison table and saves results to results.csv.

Usage:
    python noctua/evaluate.py
    python noctua/evaluate.py --checkpoint_dir checkpoints/ --dataset_base datasets/sn_explosion_hr
"""

import argparse
import csv
import os
from functools import partial

import torch
from einops import rearrange
from torch.utils.data import DataLoader

from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


# ── Model registry (must match prepare_checkpoints.py) ───────────────────────
MODEL_REGISTRY = {
    "finetune_fno.pt":           FNO,
    "finetune_tfno.pt":          TFNO,
    "finetune_unet_classic.pt":  UNetClassic,
    "finetune_unet_convnext.pt": UNetConvNext,
}

BUILD_KWARGS = {
    FNO:          dict(modes1=16, modes2=16, modes3=16, hidden_channels=128),
    TFNO:         dict(modes1=16, modes2=16, modes3=16, hidden_channels=128),
    UNetClassic:  dict(init_features=48),
    UNetConvNext: dict(init_features=42, blocks_per_stage=2),
}

COMMON_KWARGS = dict(dim_in=2, dim_out=2, n_spatial_dims=3, spatial_resolution=(64, 64, 64))
FIELD_NAMES   = ["density", "energy"]


# ── Metrics ───────────────────────────────────────────────────────────────────
def mse_per_field(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W, D, C) → (C,)"""
    return ((pred - target) ** 2).mean(dim=(0, 1, 2, 3, 4))


def rmse_per_field(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mse_per_field(pred, target).sqrt()


def rel_error_per_field(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean relative L2 error per field."""
    diff  = (pred - target).pow(2).sum(dim=(1, 2, 3, 4))       # (B, C)
    ref   = target.pow(2).sum(dim=(1, 2, 3, 4)).clamp(min=1e-8) # (B, C)
    return (diff / ref).sqrt().mean(dim=0)                       # (C,)


def spectral_error_per_field(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean relative power-spectrum error per field.
    Computed in 3-D Fourier space, averaged over radial wavenumber shells.
    pred / target: (B, 1, H, W, D, C)
    returns: (C,)
    """
    errors = []
    for c in range(pred.shape[-1]):
        p = pred[..., c].squeeze(1)      # (B, H, W, D)
        t = target[..., c].squeeze(1)
        P = torch.fft.fftn(p, dim=(-3, -2, -1)).abs() ** 2
        T = torch.fft.fftn(t, dim=(-3, -2, -1)).abs() ** 2
        rel = ((P - T).abs() / T.clamp(min=1e-8)).mean()
        errors.append(rel)
    return torch.stack(errors)


# ── Evaluation loop ───────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    agg = {k: torch.zeros(len(FIELD_NAMES)) for k in ["mse", "rmse", "rel", "spectral"]}
    n = 0

    for batch in loader:
        # Flatten time into channels: (B, T=1, H, W, D, C=2) → (B, 2, H, W, D)
        x = rearrange(batch["input_fields"], "b t ... c -> b (t c) ...").to(device)
        y_ref  = batch["output_fields"].to(device)                # (B, 1, H, W, D, 2)
        y_pred = rearrange(model(x), "b c ... -> b 1 ... c")

        agg["mse"]      += mse_per_field(y_pred, y_ref).cpu()
        agg["rmse"]     += rmse_per_field(y_pred, y_ref).cpu()
        agg["rel"]      += rel_error_per_field(y_pred, y_ref).cpu()
        agg["spectral"] += spectral_error_per_field(y_pred, y_ref).cpu()
        n += 1

    return {k: (v / n).tolist() for k, v in agg.items()}


# ── Main ──────────────────────────────────────────────────────────────────────
def run(checkpoint_dir: str, dataset_base: str, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build test dataset
    test_ds = WellDataset(
        well_base_path=os.path.dirname(dataset_base) + "/",
        well_dataset_name=os.path.basename(dataset_base),
        well_split_name="test",
        use_normalization=True,
        normalization_type=partial(ZScoreNormalization),
        n_steps_input=1,
        n_steps_output=1,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} samples")

    results = []

    for fname, model_cls in MODEL_REGISTRY.items():
        ckpt_path = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(ckpt_path):
            print(f"  Skipping {fname} — checkpoint not found at {ckpt_path}")
            continue

        print(f"\nEvaluating {fname} ...")
        ckpt  = torch.load(ckpt_path, weights_only=False, map_location=device)
        model = model_cls(**COMMON_KWARGS, **BUILD_KWARGS[model_cls]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        metrics = evaluate_model(model, loader, device)
        model_name = fname.replace("finetune_", "").replace(".pt", "")

        row = {"model": model_name}
        for field_idx, field in enumerate(FIELD_NAMES):
            row[f"MSE_{field}"]      = metrics["mse"][field_idx]
            row[f"RMSE_{field}"]     = metrics["rmse"][field_idx]
            row[f"RelErr_{field}"]   = metrics["rel"][field_idx]
            row[f"Spectral_{field}"] = metrics["spectral"][field_idx]
        results.append(row)

        print(f"  {'Field':<10} {'MSE':>12} {'RMSE':>12} {'RelErr':>10} {'Spectral':>12}")
        for field_idx, field in enumerate(FIELD_NAMES):
            print(
                f"  {field:<10}"
                f" {metrics['mse'][field_idx]:12.4e}"
                f" {metrics['rmse'][field_idx]:12.4e}"
                f" {metrics['rel'][field_idx]:10.4f}"
                f" {metrics['spectral'][field_idx]:12.4e}"
            )

    if not results:
        print("\nNo checkpoints found. Run prepare_checkpoints.py and fine-tuning first.")
        return

    # Print ranked summary (by mean RMSE across fields)
    print("\n" + "=" * 60)
    print("SUMMARY — ranked by mean RMSE")
    print("=" * 60)
    ranked = sorted(results, key=lambda r: sum(r[f"RMSE_{f}"] for f in FIELD_NAMES))
    for rank, row in enumerate(ranked, 1):
        mean_rmse = sum(row[f"RMSE_{f}"] for f in FIELD_NAMES) / len(FIELD_NAMES)
        print(f"  #{rank}  {row['model']:<20}  mean RMSE = {mean_rmse:.4e}")

    # Save CSV
    csv_path = os.path.join(checkpoint_dir, "results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results saved → {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned SN models")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument(
        "--dataset_base", type=str, default="datasets/sn_explosion_hr",
        help="Root of the sn_explosion_hr dataset",
    )
    p.add_argument("--batch_size", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        checkpoint_dir=args.checkpoint_dir,
        dataset_base=args.dataset_base,
        batch_size=args.batch_size,
    )
