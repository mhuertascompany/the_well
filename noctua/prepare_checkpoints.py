"""
Download pretrained supernova_explosion_64 weights from HuggingFace and prepare
fine-tuning checkpoints for dim_in=2, dim_out=2, 3-D 64³ grid.

The pretrained models have dim_in=24 (4 steps × 6 fields) and dim_out=6.
Our dataset has dim_in=2 (1 step × 2 fields) and dim_out=2.

Strategy:
  - Instantiate a new model with our channel dimensions
  - Copy every weight tensor whose shape matches exactly (the backbone)
  - Reinitialize IO layers that do NOT match (input projection, output projection)
  - Save in the Trainer checkpoint format so checkpoint_override can load it

Reinitialized layers per architecture:
  FNO / TFNO  : model.lifting.*   (in_channels → hidden_channels)
                model.projection.* (hidden_channels → out_channels)
  UNetClassic : encoder1.*        (dim_in conv)
                conv.*            (final dim_out conv)
  UNetConvNext: first encoder conv and final decoder conv

Usage:
    python noctua/prepare_checkpoints.py --output_dir checkpoints/
"""

import argparse
import os

import torch

from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext

# ── Target architecture ───────────────────────────────────────────────────────
DIM_IN  = 2           # 1 input step × 2 fields (density, energy)
DIM_OUT = 2           # 2 output fields
N_DIMS  = 3
RES     = (64, 64, 64)

# ── Pretrained source (supernova_explosion_64 on HuggingFace) ─────────────────
# Model IDs follow the pattern: polymathic-ai/<Architecture>-<dataset>
# Confirm availability at: https://huggingface.co/collections/polymathic-ai/the-well-benchmark-models-67e69bd7cd8e60229b5cd43e
PRETRAINED_IDS = {
    "fno":           "polymathic-ai/FNO-supernova_explosion_64",
    "tfno":          "polymathic-ai/TFNO-supernova_explosion_64",
    "unet_classic":  "polymathic-ai/UNetClassic-supernova_explosion_64",
    "unet_convnext": "polymathic-ai/UNetConvNext-supernova_explosion_64",
}

# ── Model constructors matching benchmark configs ─────────────────────────────
def build_fno():
    return FNO(
        dim_in=DIM_IN, dim_out=DIM_OUT,
        n_spatial_dims=N_DIMS, spatial_resolution=RES,
        modes1=16, modes2=16, modes3=16,
        hidden_channels=128,
    )

def build_tfno():
    return TFNO(
        dim_in=DIM_IN, dim_out=DIM_OUT,
        n_spatial_dims=N_DIMS, spatial_resolution=RES,
        modes1=16, modes2=16, modes3=16,
        hidden_channels=128,
    )

def build_unet_classic():
    return UNetClassic(
        dim_in=DIM_IN, dim_out=DIM_OUT,
        n_spatial_dims=N_DIMS, spatial_resolution=RES,
        init_features=48,
    )

def build_unet_convnext():
    return UNetConvNext(
        dim_in=DIM_IN, dim_out=DIM_OUT,
        n_spatial_dims=N_DIMS, spatial_resolution=RES,
        init_features=42, blocks_per_stage=2,
    )

BUILDERS = {
    "fno":           build_fno,
    "tfno":          build_tfno,
    "unet_classic":  build_unet_classic,
    "unet_convnext": build_unet_convnext,
}


# ── Transfer logic ────────────────────────────────────────────────────────────
def transfer_weights(pretrained: torch.nn.Module, new_model: torch.nn.Module) -> torch.nn.Module:
    """Copy all weight tensors whose shapes match; leave mismatched layers at random init."""
    pretrained_sd = pretrained.state_dict()
    new_sd        = new_model.state_dict()

    transferred = {
        k: v for k, v in pretrained_sd.items()
        if k in new_sd and v.shape == new_sd[k].shape
    }
    skipped = [k for k in pretrained_sd if k not in transferred]

    new_sd.update(transferred)
    new_model.load_state_dict(new_sd)

    frac = len(transferred) / max(len(pretrained_sd), 1) * 100
    print(f"    Transferred {len(transferred)}/{len(pretrained_sd)} tensors ({frac:.0f}%)")
    if skipped:
        print(f"    Reinitialized (shape mismatch): {skipped}")
    return new_model


def make_trainer_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save in the format expected by Trainer.load_checkpoint()."""
    optimizer = torch.optim.Adam(model.parameters())
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),           # empty — optimizer reinits from scratch
        "validation_loss": float("inf"),
        "best_validation_loss": float("inf"),
    }, path)
    print(f"    Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def prepare(output_dir: str, models: list[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name in models:
        print(f"\n[{name}]")
        hf_id   = PRETRAINED_IDS[name]
        builder = BUILDERS[name]

        # Download pretrained weights
        print(f"  Downloading {hf_id} ...")
        model_cls = {"fno": FNO, "tfno": TFNO,
                     "unet_classic": UNetClassic, "unet_convnext": UNetConvNext}[name]
        try:
            pretrained = model_cls.from_pretrained(hf_id)
        except Exception as e:
            print(f"  WARNING: could not download {hf_id}: {e}")
            print("  Falling back to random initialization (no pretrained transfer).")
            new_model = builder()
        else:
            new_model = builder()
            new_model = transfer_weights(pretrained, new_model)

        out_path = os.path.join(output_dir, f"finetune_{name}.pt")
        make_trainer_checkpoint(new_model, out_path)

    print("\nDone. Checkpoint files:")
    for f in os.listdir(output_dir):
        if f.endswith(".pt"):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
            print(f"  {f}  ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare fine-tuning checkpoints")
    p.add_argument(
        "--output_dir", type=str, default="checkpoints/",
        help="Directory to save checkpoint .pt files",
    )
    p.add_argument(
        "--models", nargs="+",
        default=["fno", "tfno", "unet_classic", "unet_convnext"],
        choices=list(BUILDERS.keys()),
        help="Which models to prepare",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(output_dir=args.output_dir, models=args.models)
