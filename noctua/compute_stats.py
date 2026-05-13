"""
Compute normalization statistics for the sn_explosion_hr dataset.

Reads all HDF5 files in datasets/sn_explosion_hr/data/train/ and writes
stats.yaml at datasets/sn_explosion_hr/stats.yaml.

stats.yaml contains per-field: mean, std, rms, mean_delta, std_delta, rms_delta.
These are required when use_normalization=True in the dataset config.

Usage:
    python noctua/compute_stats.py
    python noctua/compute_stats.py --base_path /absolute/path/to/datasets/sn_explosion_hr
"""

import argparse
import math
import os
import sys

import h5py
import torch
import yaml


# ── Custom YAML float formatter (scientific notation) ─────────────────────────
def float_representer(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.4E}")

yaml.add_representer(float, float_representer)


# ── Statistics computation ────────────────────────────────────────────────────
def compute_statistics(train_path: str, stats_path: str) -> None:
    """
    Compute per-field mean, std, rms (and delta variants) over all training files.

    Args:
        train_path:  Path to the train/ split directory containing .h5 files.
        stats_path:  Output path for stats.yaml.
    """
    if os.path.isfile(stats_path):
        print(f"stats.yaml already exists at {stats_path}. Delete it to recompute.")
        return

    h5_files = sorted(
        os.path.join(train_path, f)
        for f in os.listdir(train_path)
        if f.endswith(".h5") or f.endswith(".hdf5")
    )
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {train_path}")

    print(f"Computing statistics over {len(h5_files)} training files...")

    counts, means, variances = {}, {}, {}
    counts_delta, means_delta, variances_delta = {}, {}, {}

    for path in h5_files:
        with h5py.File(path, "r") as f:
            for i in range(3):
                ti = f"t{i}_fields"
                for field in f[ti].attrs["field_names"]:
                    data = torch.as_tensor(f[ti][field][:], dtype=torch.float64)
                    count = math.prod(data.shape[: data.ndim - i])
                    var, mean = torch.var_mean(
                        data,
                        dim=tuple(range(0, data.ndim - i)),
                        unbiased=False,
                    )
                    counts.setdefault(field, []).append(count)
                    means.setdefault(field, []).append(mean)
                    variances.setdefault(field, []).append(var)

                    if f[ti][field].attrs["time_varying"]:
                        delta = data[:, 1:] - data[:, :-1]
                        count_d = math.prod(delta.shape[: delta.ndim - i])
                        var_d, mean_d = torch.var_mean(
                            delta,
                            dim=tuple(range(0, delta.ndim - i)),
                            unbiased=False,
                        )
                        counts_delta.setdefault(field, []).append(count_d)
                        means_delta.setdefault(field, []).append(mean_d)
                        variances_delta.setdefault(field, []).append(var_d)

    out_means, out_stds, out_rmss = {}, {}, {}
    out_means_d, out_stds_d, out_rmss_d = {}, {}, {}

    for field in counts:
        w = torch.tensor(counts[field], dtype=torch.float64)
        w = w / w.sum()
        ms = torch.stack(means[field])
        vs = torch.stack(variances[field])

        m1 = torch.einsum("i...,i", ms, w)
        m2 = torch.einsum("i...,i", vs + ms ** 2, w)
        std = (m2 - m1 ** 2).sqrt()
        rms = m2.sqrt()

#        assert torch.all(std > 1e-4), (
#            f"Standard deviation of '{field}' is abnormally low ({std}). "
#            "Check that this field has physical variation in the training set."
#        )

        out_means[field] = m1.tolist()
        out_stds[field]  = std.tolist()
        out_rmss[field]  = rms.tolist()

        if field in counts_delta:
            wd = torch.tensor(counts_delta[field], dtype=torch.float64)
            wd = wd / wd.sum()
            msd = torch.stack(means_delta[field])
            vsd = torch.stack(variances_delta[field])

            m1d = torch.einsum("i...,i", msd, wd)
            m2d = torch.einsum("i...,i", vsd + msd ** 2, wd)
            stdd = (m2d - m1d ** 2).sqrt()
            rmsd = m2d.sqrt()

            out_means_d[field] = m1d.tolist()
            out_stds_d[field]  = stdd.tolist()
            out_rmss_d[field]  = rmsd.tolist()

    stats = {
        "mean":       out_means,
        "std":        out_stds,
        "rms":        out_rmss,
        "mean_delta": out_means_d,
        "std_delta":  out_stds_d,
        "rms_delta":  out_rmss_d,
    }

    with open(stats_path, "w", encoding="utf8") as f:
        yaml.dump(stats, f)

    print(f"Saved stats.yaml → {stats_path}")
    print("Field statistics:")
    for field in out_means:
        print(f"  {field:12s}  mean={out_means[field]:.4E}  std={out_stds[field]:.4E}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Well normalization statistics")
    p.add_argument(
        "--base_path",
        type=str,
        default="datasets/sn_explosion_hr",
        help="Root of the sn_explosion_hr dataset (contains data/ and where stats.yaml will be written)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_statistics(
        train_path=os.path.join(args.base_path, "data", "train"),
        stats_path=os.path.join(args.base_path, "stats.yaml"),
    )
