#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end SN fine-tuning pipeline on Noctua
#
# Steps:
#   1. Convert raw simulation HDF5 files to Well format
#   2. Compute normalization statistics
#   3. Download pretrained weights and prepare checkpoints
#   4. Fine-tune each model
#
# Edit the CONFIGURATION block below before running.
# Run from the repository root:  bash noctua/run_pipeline.sh
# =============================================================================
set -euo pipefail

# ── CONFIGURATION — fill in before running ────────────────────────────────────
SIM_DIR="__PLACEHOLDER__/simulations"   # directory containing raw .h5 sim files
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_BASE="${REPO_ROOT}/datasets/sn_explosion_hr"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
LR="5e-5"                               # fine-tuning learning rate
EPOCHS=100                              # override trainer epochs if needed
SERVER="local"                          # "local" or "noctua" (see configs/server/)
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo " SN Explosion Fine-tuning Pipeline"
echo " Repo root  : ${REPO_ROOT}"
echo " Sim dir    : ${SIM_DIR}"
echo " Dataset    : ${DATASET_BASE}"
echo " Checkpoints: ${CHECKPOINT_DIR}"
echo "================================================================"

# ── Step 1: Convert data ──────────────────────────────────────────────────────
echo ""
echo "[1/4] Converting simulation files to Well HDF5 format..."
python "${REPO_ROOT}/noctua/convert_data.py" \
    --input_dir   "${SIM_DIR}" \
    --output_dir  "${DATASET_BASE}/data" \
    --train_frac  0.8 \
    --val_frac    0.1 \
    --trajs_per_file 4 \
    --seed 42

# ── Step 2: Compute statistics ────────────────────────────────────────────────
echo ""
echo "[2/4] Computing normalization statistics..."
python "${REPO_ROOT}/noctua/compute_stats.py" \
    --base_path "${DATASET_BASE}"

# ── Step 3: Prepare checkpoints ───────────────────────────────────────────────
echo ""
echo "[3/4] Preparing pretrained checkpoints..."
python "${REPO_ROOT}/noctua/prepare_checkpoints.py" \
    --output_dir "${CHECKPOINT_DIR}" \
    --models fno tfno unet_classic unet_convnext

# ── Step 4: Fine-tune each model ──────────────────────────────────────────────
echo ""
echo "[4/4] Fine-tuning models..."
cd "${REPO_ROOT}/the_well/benchmark"

MODELS=(
    "finetune_fno_sn:${CHECKPOINT_DIR}/finetune_fno.pt"
    "finetune_tfno_sn:${CHECKPOINT_DIR}/finetune_tfno.pt"
    "finetune_unet_classic_sn:${CHECKPOINT_DIR}/finetune_unet_classic.pt"
    "finetune_unet_convnext_sn:${CHECKPOINT_DIR}/finetune_unet_convnext.pt"
)

for entry in "${MODELS[@]}"; do
    EXPERIMENT="${entry%%:*}"
    CHECKPOINT="${entry##*:}"
    echo ""
    echo "  Training: experiment=${EXPERIMENT}"
    python train.py \
        experiment="${EXPERIMENT}" \
        server="${SERVER}" \
        "optimizer.lr=${LR}" \
        "trainer.epochs=${EPOCHS}" \
        "checkpoint_override=${CHECKPOINT}"
done

echo ""
echo "================================================================"
echo " All models trained. Evaluate with:"
echo "   python ${REPO_ROOT}/noctua/evaluate.py"
echo "================================================================"
