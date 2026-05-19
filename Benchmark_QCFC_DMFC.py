#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import kpe_config as cfg

NO_GSR_DIR = cfg.ANATOMICAL_TS_DIR
GSR_DIR = cfg.GLOBAL_TS_DIR
NO_GSR_SCRUB_CSV = cfg.ANATOMICAL_SCRUB_CSV
GSR_SCRUB_CSV = cfg.GLOBAL_SCRUB_CSV
OUT_DIR = (cfg.RESULTS_DIR / "qc_fc_dm_fc_results")

ATLAS_TAG = "schaefer400"         # "tian_s2" / "schaefer400"
CENTROIDS_FILE = ( cfg.BASE_DIR / "atlas_centroids" / "schaefer400_centroids.csv")

# Motion column from scrubbing report
FD_COLUMN = "fd_mean_filtered"      #fd_mean_raw   / fc_mean_filtered

ALPHA = 0.05

def read_timeseries_csv(path: Path, expected_nodes: int | None = None):
    """
    Returns:
      arr shape = n_timepoints x n_nodes

    Assumes atlas node count should stay fixed across all files.
    Does NOT silently drop columns, because that breaks edge comparability.
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()

    if numeric.empty:
        raw = pd.read_csv(path, header=None)
        numeric = raw.select_dtypes(include=[np.number]).copy()

    arr = numeric.to_numpy(dtype=float)

    if arr.ndim != 2 or min(arr.shape) < 2:
        raise ValueError(f"Bad timeseries shape in {path}: {arr.shape}")

    # Common case: timepoints > parcels
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T

    n_timepoints, n_nodes = arr.shape

    if expected_nodes is not None and n_nodes != expected_nodes:
        raise ValueError(
            f"Node count mismatch in {path.name}: got {n_nodes}, expected {expected_nodes}"
        )

    # Reject files with non-finite values instead of dropping columns
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path.name}")

    # Reject files with zero-variance nodes instead of dropping them
    bad_zero_var = np.where(np.nanstd(arr, axis=0) == 0)[0]
    if len(bad_zero_var) > 0:
        raise ValueError(
            f"Zero-variance nodes found in {path.name}: n_bad={len(bad_zero_var)}"
        )

    if n_nodes < 2:
        raise ValueError(f"Need at least 2 valid nodes in {path.name}")

    return arr


def corr_upper_triangle(ts):
    """
    ts: n_timepoints x n_nodes
    returns:
      edge vector, upper triangle indices
    """
    cmat = np.corrcoef(ts, rowvar=False)
    iu = np.triu_indices(cmat.shape[0], k=1)
    return cmat[iu], iu


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)

    valid = np.isfinite(pvals)
    pv = pvals[valid]
    m = len(pv)

    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out


def load_centroids(path: Path):
    df = pd.read_csv(path)
    required = {"node", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"Centroids file must contain columns: {required}")

    df = df.sort_values("node").reset_index(drop=True)
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    return coords


def compute_edge_distances(coords):
    """
    coords shape: n_nodes x 3
    returns upper-triangle distance vector
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=2))
    iu = np.triu_indices(dist_mat.shape[0], k=1)
    return dist_mat[iu]


def load_scrubbing_report(scrub_csv: Path, dataset_name: str, fd_column: str):
    """
    Reads your scrubbing report and returns one row per run with motion summary.
    Your scrubbing file includes columns such as:
    subject, session, task, acq, run, fd_mean_raw, fd_mean_filtered,
    fd_max_filtered, scrubbed_volumes, scrub_percent, high_motion_skip, gsr_applied
    """
    df = pd.read_csv(scrub_csv)

    required = ["subject", "session", "task", "run", fd_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{scrub_csv} is missing columns: {missing}")

    if "acq" not in df.columns:
        df["acq"] = ""
    df["acq"] = df["acq"].fillna("").astype(str)

    optional_cols = []
    for c in ["scrubbed_volumes", "scrub_percent", "high_motion_skip", "gsr_applied",
              "fd_mean_raw", "fd_mean_filtered", "fd_max_filtered",
              "raw_spikes", "filtered_spikes", "total_vols",
              "dvars_post", "confound_var_explained"]:
        if c in df.columns:
            optional_cols.append(c)

    out = df[["subject", "session", "task", "acq", "run"] + optional_cols].copy()
    out = out.rename(columns={fd_column: "mean_fd"})

    out["dataset_name"] = dataset_name
    out["subject"] = out["subject"].astype(str)
    out["session"] = out["session"].astype(str)

    # The CSV has task like "rest", while filenames often have "task-rest"
    out["task"] = out["task"].astype(str).apply(
        lambda x: x if x.startswith("task-") else f"task-{x}"
    )

    out["run"] = out["run"].astype(str)

    return out


# =========================================================
# BUILD SCAN-LEVEL TABLE
# =========================================================

def build_dataset_table(dataset_name: str, ts_root: Path, scrub_csv: Path, atlas_tag: str, fd_column: str):
    ts_files = cfg.find_ts_files(ts_root, atlas_tag)
    if not ts_files:
        raise FileNotFoundError(f"No *{atlas_tag}_ts.csv files found in {ts_root}")

    rows = []
    expected_edges = None
    expected_nodes = None

    bad_files = []

    for ts_path in ts_files:
        ent = cfg.parse_entities(ts_path.name)
        if ent.get("subject") is None:
            bad_files.append((ts_path.name, "Could not parse subject"))
            continue

        try:
            ts = read_timeseries_csv(ts_path, expected_nodes=expected_nodes)
            edge_vec, _ = corr_upper_triangle(ts)

            if expected_edges is None:
                expected_edges = len(edge_vec)
                expected_nodes = ts.shape[1]
            elif len(edge_vec) != expected_edges:
                bad_files.append(
                    (ts_path.name,
                     f"Edge mismatch: expected {expected_edges}, got {len(edge_vec)}")
                )
                continue

        except Exception as e:
            bad_files.append((ts_path.name, str(e)))
            continue

        row = {
            "dataset_name": dataset_name,
            "subject": ent.get("subject"),
            "session": ent.get("session"),
            "task": ent.get("task"),
            "run": ent.get("run"),
            "acq": ent.get("acq", "") or "",
            "ts_path": str(ts_path),
            "n_timepoints": ts.shape[0],
            "n_nodes": ts.shape[1],
        }

        for i, v in enumerate(edge_vec):
            row[f"edge_{i:06d}"] = float(v)

        rows.append(row)

    if bad_files:
        print(f"\nWARNING: skipped {len(bad_files)} bad files in {dataset_name}:")
        for name, reason in bad_files[:20]:
            print(f"  {name} -> {reason}")

    ts_df = pd.DataFrame(rows)

    scrub_df = load_scrubbing_report(scrub_csv, dataset_name, fd_column)

    merged = ts_df.merge(
        scrub_df,
        on=["dataset_name", "subject", "session", "task", "acq", "run"],
        how="left"
    )

    if merged["mean_fd"].isna().any():
        missing = merged.loc[
            merged["mean_fd"].isna(),
            ["subject", "session", "task", "acq", "run", "ts_path"]
        ]
        print("\nWARNING: some time-series files did not match the scrubbing report:")
        print(missing.head(20).to_string(index=False))

    merged = merged[merged["mean_fd"].notna()].copy()

    if merged.empty:
        raise ValueError(f"No scans left after merging with {scrub_csv}")

    return merged, expected_nodes, expected_edges


# =========================================================
# QC-FC
# =========================================================

def compute_qc_fc(edge_df: pd.DataFrame, dataset_name: str):
    """
    Simplified QC-FC:
    correlation between mean FD and each edge across scans.
    """
    d = edge_df[edge_df["dataset_name"] == dataset_name].copy()
    edge_cols = [c for c in d.columns if c.startswith("edge_")]
    motion = d["mean_fd"].to_numpy(dtype=float)

    results = []
    for edge in edge_cols:
        y = d[edge].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(motion)
        n = int(valid.sum())

        if n < 4:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(motion[valid], y[valid])

        results.append({
            "edge": edge,
            "qc_fc_r": r,
            "p_unc": p,
            "n": n,
        })

    res = pd.DataFrame(results)
    res["p_fdr"] = fdr_bh(res["p_unc"].to_numpy())

    summary = pd.DataFrame([{
        "dataset_name": dataset_name,
        "atlas_tag": ATLAS_TAG,
        "fd_column": FD_COLUMN,
        "n_scans": int(len(d)),
        "n_subjects": int(d["subject"].nunique()),
        "n_edges": int(len(res)),
        "pct_sig_unc_p_lt_0_05": float(100 * np.nanmean(res["p_unc"] < ALPHA)),
        "pct_sig_fdr_q_lt_0_05": float(100 * np.nanmean(res["p_fdr"] < ALPHA)),
        "median_abs_qc_fc": float(np.nanmedian(np.abs(res["qc_fc_r"]))),
        "mean_abs_qc_fc": float(np.nanmean(np.abs(res["qc_fc_r"]))),
        "mean_fd_mean": float(np.nanmean(d["mean_fd"])),
        "mean_fd_sd": float(np.nanstd(d["mean_fd"], ddof=1)),
    }])

    return res, summary


# =========================================================
# DM-FC
# =========================================================

def compute_dm_fc(qc_fc_df: pd.DataFrame, edge_distances: np.ndarray, dataset_name: str):
    """
    DM-FC:
    correlation between edge distance and QC-FC values.
    """
    if len(qc_fc_df) != len(edge_distances):
        raise ValueError(
            f"DM-FC mismatch: {len(qc_fc_df)} QC-FC edges vs {len(edge_distances)} distances"
        )

    qc = qc_fc_df["qc_fc_r"].to_numpy(dtype=float)
    dist = np.asarray(edge_distances, dtype=float)

    valid = np.isfinite(qc) & np.isfinite(dist)
    n = int(valid.sum())

    if n < 4:
        r, p = np.nan, np.nan
    else:
        r, p = stats.pearsonr(dist[valid], qc[valid])

    summary = pd.DataFrame([{
        "dataset_name": dataset_name,
        "atlas_tag": ATLAS_TAG,
        "fd_column": FD_COLUMN,
        "dm_fc_r": float(r) if np.isfinite(r) else np.nan,
        "abs_dm_fc_r": float(abs(r)) if np.isfinite(r) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
        "n_edges_used": n,
    }])

    return summary


# =========================================================
# MAIN
# =========================================================

def main():
    cfg.ensure_dir(OUT_DIR)

    print(f"Running atlas: {ATLAS_TAG}")
    print(f"Using motion column: {FD_COLUMN}")

    print("Loading atlas centroids...")
    coords = load_centroids(CENTROIDS_FILE)
    edge_distances = compute_edge_distances(coords)

    print("Building no-GSR dataset...")
    no_gsr_df, no_gsr_nodes, no_gsr_edges = build_dataset_table(
        dataset_name="no_gsr",
        ts_root=NO_GSR_DIR,
        scrub_csv=NO_GSR_SCRUB_CSV,
        atlas_tag=ATLAS_TAG,
        fd_column=FD_COLUMN,
    )

    print("Building GSR dataset...")
    gsr_df, gsr_nodes, gsr_edges = build_dataset_table(
        dataset_name="gsr",
        ts_root=GSR_DIR,
        scrub_csv=GSR_SCRUB_CSV,
        atlas_tag=ATLAS_TAG,
        fd_column=FD_COLUMN,
    )

    if no_gsr_nodes != gsr_nodes or no_gsr_edges != gsr_edges:
        raise ValueError("no_gsr and gsr do not have matching atlas dimensionality")

    if len(edge_distances) != no_gsr_edges:
        raise ValueError(
            f"Centroids imply {len(edge_distances)} edges, but timeseries imply {no_gsr_edges} edges"
        )

    all_df = pd.concat([no_gsr_df, gsr_df], ignore_index=True)

    run_out = OUT_DIR / f"{ATLAS_TAG}_{FD_COLUMN}"
    cfg.ensure_dir(run_out)
    all_df.to_csv(run_out / "all_scan_level_edge_data.csv", index=False)

    all_qc_summaries = []
    all_dmfc_summaries = []

    for dataset_name in ["no_gsr", "gsr"]:
        print(f"\nComputing QC-FC for {dataset_name}...")
        qc_df, qc_summary = compute_qc_fc(all_df, dataset_name)

        ds_out = run_out / dataset_name
        cfg.ensure_dir(ds_out)

        qc_df.to_csv(ds_out / "edgewise_qc_fc.csv", index=False)
        qc_summary.to_csv(ds_out / "qc_fc_summary.csv", index=False)

        print(f"Computing DM-FC for {dataset_name}...")
        dmfc_summary = compute_dm_fc(qc_df, edge_distances, dataset_name)
        dmfc_summary.to_csv(ds_out / "dm_fc_summary.csv", index=False)

        # also save QC-FC merged with distances for inspection
        qc_with_dist = qc_df.copy()
        qc_with_dist["edge_distance"] = edge_distances
        qc_with_dist.to_csv(ds_out / "edgewise_qc_fc_with_distance.csv", index=False)

        all_qc_summaries.append(qc_summary)
        all_dmfc_summaries.append(dmfc_summary)

        print("\n" + "=" * 60)
        print(dataset_name.upper())
        print("QC-FC summary:")
        print(qc_summary.to_string(index=False))
        print("\nDM-FC summary:")
        print(dmfc_summary.to_string(index=False))
        print("=" * 60)

    summary_all = pd.concat(all_qc_summaries, ignore_index=True)
    summary_all.to_csv(run_out / "qc_fc_summary_all_datasets.csv", index=False)

    dmfc_all = pd.concat(all_dmfc_summaries, ignore_index=True)
    dmfc_all.to_csv(run_out / "dm_fc_summary_all_datasets.csv", index=False)

    print(f"\nDone. Results saved to: {run_out}")


if __name__ == "__main__":
    main()