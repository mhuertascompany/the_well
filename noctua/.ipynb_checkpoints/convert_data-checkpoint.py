"""
Convert SN explosion simulation files to the Well HDF5 format.

Input format (one file per simulation pair):
    input (density, energy) at t0 and truth (density, energy) at t0 + 2 Myr

Output format (Well-compatible HDF5):
    Groups N_TRAJ_PER_FILE simulation pairs into one file.
    Each file contains two t0_fields ("density", "energy") with shape
    (N_traj, T=2, 64, 64, 64), where T=0 is t0 and T=1 is t0+20 Myr.

Usage:
    python noctua/convert_data.py \
        --input_dir  /PATH/TO/YOUR/SIMULATIONS \
        --output_dir datasets/sn_explosion_hr/data \
        --train_frac 0.8 \
        --val_frac   0.1 \
        --trajs_per_file 4 \
        --seed 42
"""

import argparse
import os
import random

import h5py
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME    = "sn_explosion_hr"
FIELD_NAMES     = ["density", "energy"]   # order matches axis-0 of "raw"/"label"
SPATIAL_DIMS    = ["x", "y", "z"]
GRID_SIZE       = 64
N_TIME_STEPS    = 2                       # t0 and t0 + 20 Myr
TIME_VALUES     = [0.0, 2.0]            # Myr


# ── Core writer ───────────────────────────────────────────────────────────────
def write_well_hdf5(output_path: str, sim_files: list[str]) -> None:
    """
    Read a list of raw simulation files and write one Well-compatible HDF5 file.

    Args:
        output_path: Destination .h5 path.
        sim_files:   List of input HDF5 paths to bundle into this file.
                     Each contributes one trajectory.
    """
    N = len(sim_files)
    G = GRID_SIZE

    # Pre-allocate arrays: (N_traj, T, 64, 64, 64)
    density = np.empty((N, N_TIME_STEPS, G, G, G), dtype=np.float32)
    energy  = np.empty((N, N_TIME_STEPS, G, G, G), dtype=np.float32)
    boxsize = np.empty(N, dtype=np.float32)
    central_density = np.empty(N, dtype=np.float32)

    for i, sim_path in enumerate(sim_files):
        with h5py.File(sim_path, "r") as src:
            densitybefore = np.log(src["3DGridBefore"]['GridDensity[g.cm-3]'][:])    # (64, 64, 64)
            energybefore = np.log(src["3DGridBefore"]['GridEnergy[(cm per s)^2]'][:])    # (64, 64, 64)
            densityafter = np.log(src["3DGridAfter"]['GridDensity[g.cm-3]'][:])    # (64, 64, 64)
            energyafter = np.log(src["3DGridAfter"]['GridEnergy[(cm per s)^2]'][:])    # (64, 64, 64)
            # Clean
            densitybefore[~np.isfinite(densitybefore)] = 0
            energybefore[~np.isfinite(energybefore)] = 0
            densityafter[~np.isfinite(densityafter)] = 0
            energyafter[~np.isfinite(energyafter)] = 0
            box = src['SupernovaInfo'].attrs['BoxSize[kpc]']
            central_dens = src['SupernovaInfo'].attrs['CentralDensity[g per cm3]']
        # axis-0 of raw/label: [density=0, energy=1]
        density[i, 0] = densitybefore    # density  at t0
        density[i, 1] = densityafter  # density  at t0 + 20 Myr
        energy[i, 0]  = energybefore    # energy   at t0
        energy[i, 1]  = energyafter  # energy   at t0 + 20 Myr
        boxsize[i] = box          # boxsize
        central_density[i] = central_dens   # central density

    coords  = np.linspace(0.0, 1.0, G, dtype=np.float32)
    t_axis  = np.array(TIME_VALUES, dtype=np.float32)
    open_mask = np.zeros(G, dtype=bool)

    with h5py.File(output_path, "w") as f:
        # ── Top-level attributes ──────────────────────────────────────────
        f.attrs["dataset_name"]          = DATASET_NAME
        f.attrs["grid_type"]             = "cartesian"
        f.attrs["n_spatial_dims"]        = 3
        f.attrs["n_trajectories"]        = N
        f.attrs["simulation_parameters"] = []

        # ── Boundary conditions (open on all axes) ────────────────────────
        bc = f.create_group("boundary_conditions")
        for ax in SPATIAL_DIMS:
            sg = bc.create_group(f"{ax}_open")
            sg.attrs["associated_dims"]   = [ax]
            sg.attrs["associated_fields"] = []
            sg.attrs["bc_type"]           = "OPEN"
            sg.attrs["sample_varying"]    = False
            sg.attrs["time_varying"]      = False
            sg.create_dataset("mask", data=open_mask)

        # ── Dimensions ────────────────────────────────────────────────────
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = SPATIAL_DIMS
        dims.create_dataset("time", data=t_axis)
        dims["time"].attrs["sample_varying"] = False
        for ax in SPATIAL_DIMS:
            dims.create_dataset(ax, data=coords)
            dims[ax].attrs["sample_varying"] = False

        # ── Scalars (none for this dataset) ──────────────────────────────
        sc = f.create_group("scalars")
        sc.attrs["field_names"] = ["Boxsize", "Central density"]
        ds = sc.create_dataset('Boxsize', data=boxsize)
        ds.attrs["sample_varying"] = True
        ds.attrs["time_varying"]   = False
        ds = sc.create_dataset('Central density', data=central_density)
        ds.attrs["sample_varying"] = True
        ds.attrs["time_varying"]   = False

        # ── t0_fields: density and energy ─────────────────────────────────
        # Shape stored: (N_traj, T=2, 64, 64, 64)
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = FIELD_NAMES
        for name, arr in [("density", density), ("energy", energy)]:
            ds = t0.create_dataset(name, data=arr)
            ds.attrs["dim_varying"]    = [True, True, True]
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"]   = True

        # ── Required empty groups ─────────────────────────────────────────
        f.create_group("t1_fields").attrs["field_names"] = []
        f.create_group("t2_fields").attrs["field_names"] = []


# ── Split & write ─────────────────────────────────────────────────────────────
def convert(
    input_dir: str,
    output_dir: str,
    train_frac: float,
    val_frac: float,
    trajs_per_file: int,
    seed: int,
) -> None:
    """Discover all simulation files, split, and write Well HDF5 files."""
    # Discover all HDF5 files in input_dir
    all_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    )
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    print(f"Found {len(all_files)} simulation files in {input_dir}")

    # Shuffle deterministically then split
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    n_test  = n_total - n_train - n_val

    splits = {
        "train": all_files[:n_train],
        "valid": all_files[n_train : n_train + n_val],
        "test":  all_files[n_train + n_val :],
    }

    print(f"Split: train={n_train}  valid={n_val}  test={n_test}")

    for split, files in splits.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Chunk into groups of trajs_per_file
        chunks = [
            files[i : i + trajs_per_file]
            for i in range(0, len(files), trajs_per_file)
        ]
        for file_idx, chunk in enumerate(chunks):
            out_path = os.path.join(split_dir, f"sn_{file_idx:04d}.h5")
            write_well_hdf5(out_path, chunk)
            print(f"  [{split}] {file_idx+1}/{len(chunks)} → {out_path}  ({len(chunk)} trajs)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert SN simulation HDF5 files to Well format")
    p.add_argument(
        "--input_dir",
        type=str,
        default="__PLACEHOLDER__/simulations",
        help="Directory containing raw simulation HDF5 files (one per simulation pair)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="datasets/sn_explosion_hr/data",
        help="Root output directory; train/valid/test subdirs are created automatically",
    )
    p.add_argument("--train_frac",      type=float, default=0.8)
    p.add_argument("--val_frac",        type=float, default=0.1)
    p.add_argument("--trajs_per_file",  type=int,   default=4,
                   help="Number of trajectories bundled into each output HDF5 file")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        trajs_per_file=args.trajs_per_file,
        seed=args.seed,
    )
