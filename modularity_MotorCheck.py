from pathlib import Path
import re

import numpy as np
import pandas as pd
import networkx as nx

import kpe_config as cfg


# ============================================================
# USER SETTINGS
# ============================================================

# Choose:
# ["tau"]
# ["yale"]
# ["tau", "yale"]
DATASETS_TO_RUN = ["tau", "yale"]

# Usually keep both, because this script compares anatomical vs global
PIPELINES_TO_RUN = ["anatomical", "global"]

ATLAS_TAG = "schaefer400"

ZERO_NEGATIVE_EDGES_FOR_MODULARITY = True

OUT_DIR = cfg.RESULTS_DIR / "motor_check_results"
cfg.ensure_dir(OUT_DIR)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_task(value):
    value = str(value)

    if value.startswith("task-"):
        return value

    return f"task-{value}"


def normalize_run(value):
    value = str(value)

    if value.startswith("run-"):
        return value

    return f"run-{value}"


def normalize_acq(value):
    if pd.isna(value):
        return ""

    return str(value)


def load_numeric_ts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Background" in df.columns:
        df = df.drop(columns=["Background"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    return df


def compute_fc(ts_df: pd.DataFrame) -> np.ndarray:
    ts = ts_df.values

    corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    np.fill_diagonal(corr, 1.0)

    return corr


def compute_modularity_q(fc: np.ndarray) -> float:
    """
    Louvain modularity on a positive-weight graph.

    Negative edges are zeroed by default because standard Louvain modularity is
    easiest to interpret with positive weights.
    """

    mat = fc.copy()
    np.fill_diagonal(mat, 0.0)

    if ZERO_NEGATIVE_EDGES_FOR_MODULARITY:
        mat[mat < 0] = 0.0

    G = nx.from_numpy_array(mat)

    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return np.nan

    communities = nx.community.louvain_communities(
        G,
        weight="weight",
        seed=42,
    )

    q = nx.community.modularity(
        G,
        communities,
        weight="weight",
    )

    return float(q)


# ============================================================
# MOTOR CHECK
# ============================================================

def find_schaefer_sommot_columns(columns):
    """
    Uses the Schaefer 7-network naming convention:
        7Networks_LH_SomMot_*
        7Networks_RH_SomMot_*
    """

    left_cols = [c for c in columns if "LH_SomMot" in c]
    right_cols = [c for c in columns if "RH_SomMot" in c]

    return left_cols, right_cols


def compute_motor_lr_schaefer(ts_df: pd.DataFrame):
    """
    Motor sanity check:
    left SomMot mean signal vs right SomMot mean signal.
    """

    left_cols, right_cols = find_schaefer_sommot_columns(ts_df.columns.tolist())

    if len(left_cols) == 0 or len(right_cols) == 0:
        return {
            "motor_lr_corr": np.nan,
            "motor_n_left": len(left_cols),
            "motor_n_right": len(right_cols),
            "motor_method": "SomMot columns not found",
        }

    left_mean = ts_df[left_cols].mean(axis=1)
    right_mean = ts_df[right_cols].mean(axis=1)

    good = np.isfinite(left_mean.values) & np.isfinite(right_mean.values)

    if good.sum() < 3:
        return {
            "motor_lr_corr": np.nan,
            "motor_n_left": len(left_cols),
            "motor_n_right": len(right_cols),
            "motor_method": "insufficient_data",
        }

    r = np.corrcoef(left_mean.values[good], right_mean.values[good])[0, 1]

    return {
        "motor_lr_corr": float(r),
        "motor_n_left": len(left_cols),
        "motor_n_right": len(right_cols),
        "motor_method": "mean(LH_SomMot) vs mean(RH_SomMot)",
    }


# ============================================================
# FILE PROCESSING
# ============================================================

def process_one_file(csv_path: Path, atlas_tag: str):
    ts_df = load_numeric_ts(csv_path)
    meta = cfg.parse_entities(csv_path.name)
    meta["file"] = csv_path.name

    row = {
        **meta,
        "atlas_tag": atlas_tag,
        "n_timepoints": ts_df.shape[0],
        "n_rois": ts_df.shape[1],
    }

    if ts_df.shape[0] < 3 or ts_df.shape[1] < 2:
        row.update({
            "modularity_q": np.nan,
            "motor_lr_corr": np.nan,
            "motor_n_left": np.nan,
            "motor_n_right": np.nan,
            "motor_method": "insufficient_data",
        })
        return row

    fc = compute_fc(ts_df)

    row["modularity_q"] = compute_modularity_q(fc)

    if "schaefer" in atlas_tag.lower():
        motor = compute_motor_lr_schaefer(ts_df)
    else:
        motor = {
            "motor_lr_corr": np.nan,
            "motor_n_left": np.nan,
            "motor_n_right": np.nan,
            "motor_method": "not computed for this atlas",
        }

    row.update(motor)

    return row


# ============================================================
# SCRUBBING REPORT
# ============================================================

def load_scrub_report(scrub_csv: Path):
    if scrub_csv is None or not scrub_csv.exists():
        print(f"WARNING: scrubbing report not found: {scrub_csv}")
        return pd.DataFrame()

    df = pd.read_csv(scrub_csv)

    keep_cols = [
        c for c in [
            "subject",
            "session",
            "task",
            "acq",
            "run",
            "fd_mean_raw",
            "fd_mean_filtered",
            "fd_max_filtered",
            "raw_spikes",
            "filtered_spikes",
            "scrubbed_volumes",
            "scrub_percent",
            "high_motion_skip",
            "gsr_applied",
        ]
        if c in df.columns
    ]

    df = df[keep_cols].copy()

    if "acq" not in df.columns:
        df["acq"] = ""

    df["subject"] = df["subject"].astype(str)
    df["session"] = df["session"].astype(str)
    df["task"] = df["task"].apply(normalize_task)
    df["run"] = df["run"].apply(normalize_run)
    df["acq"] = df["acq"].apply(normalize_acq)

    return df


# ============================================================
# BUILD METRICS
# ============================================================

def build_metrics_for_dir(
    dataset_name: str,
    pipeline_name: str,
    ts_dir: Path,
    scrub_csv: Path,
    atlas_tag: str,
):
    print("\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Time-series folder: {ts_dir}")
    print(f"Scrubbing file: {scrub_csv}")
    print("==============================")

    files = cfg.find_ts_files(ts_dir, atlas_tag)

    rows = []

    for f in files:
        try:
            row = process_one_file(f, atlas_tag)

            row["dataset"] = dataset_name
            row["pipeline"] = pipeline_name

            row["subject"] = str(row.get("subject", ""))
            row["session"] = str(row.get("session", ""))
            row["task"] = normalize_task(row.get("task", ""))
            row["run"] = normalize_run(row.get("run", ""))
            row["acq"] = normalize_acq(row.get("acq", ""))

            rows.append(row)

        except Exception as e:
            print(f"[{dataset_name} | {pipeline_name}] ERROR in {f.name}: {e}")

    metrics_df = pd.DataFrame(rows)

    if metrics_df.empty:
        print(f"WARNING: no valid files for {dataset_name} | {pipeline_name}")
        return metrics_df

    scrub_df = load_scrub_report(scrub_csv)

    if not scrub_df.empty:
        metrics_df = metrics_df.merge(
            scrub_df,
            on=["subject", "session", "task", "acq", "run"],
            how="left",
        )

    return metrics_df


# ============================================================
# MAIN
# ============================================================

def main():
    all_results = []

    for dataset_name in DATASETS_TO_RUN:

        dataset_results = []

        for pipeline_name in PIPELINES_TO_RUN:
            ts_dir = cfg.DATASETS[dataset_name][pipeline_name]
            scrub_csv = cfg.DATASETS[dataset_name][f"{pipeline_name}_scrub"]

            df = build_metrics_for_dir(
                dataset_name=dataset_name,
                pipeline_name=pipeline_name,
                ts_dir=ts_dir,
                scrub_csv=scrub_csv,
                atlas_tag=ATLAS_TAG,
            )

            if not df.empty:
                dataset_results.append(df)
                all_results.append(df)

        if dataset_results:
            dataset_combined = pd.concat(dataset_results, ignore_index=True)

            dataset_out_dir = OUT_DIR / dataset_name
            cfg.ensure_dir(dataset_out_dir)

            out_csv = dataset_out_dir / f"motor_check_comparison_{dataset_name}_{ATLAS_TAG}.csv"
            dataset_combined.to_csv(out_csv, index=False)

            key_cols = ["subject", "session", "task", "run", "atlas_tag"]

            wide = dataset_combined.pivot_table(
                index=key_cols,
                columns="pipeline",
                values=[
                    "modularity_q",
                    "motor_lr_corr",
                    "fd_mean_filtered",
                    "scrub_percent",
                ],
                aggfunc="first",
            )

            wide.columns = [f"{a}_{b}" for a, b in wide.columns]
            wide = wide.reset_index()

            if (
                "modularity_q_anatomical" in wide.columns
                and "modularity_q_global" in wide.columns
            ):
                wide["delta_modularity_global_minus_anatomical"] = (
                    wide["modularity_q_global"] - wide["modularity_q_anatomical"]
                )

            if (
                "motor_lr_corr_anatomical" in wide.columns
                and "motor_lr_corr_global" in wide.columns
            ):
                wide["delta_motor_global_minus_anatomical"] = (
                    wide["motor_lr_corr_global"] - wide["motor_lr_corr_anatomical"]
                )

            wide_csv = dataset_out_dir / f"motor_check_comparison_{dataset_name}_{ATLAS_TAG}_wide.csv"
            wide.to_csv(wide_csv, index=False)

            print("\n===== SUMMARY =====")
            print(f"Dataset: {dataset_name}")

            for metric in ["modularity_q", "motor_lr_corr"]:
                for pipeline in PIPELINES_TO_RUN:
                    vals = dataset_combined.loc[
                        dataset_combined["pipeline"] == pipeline,
                        metric,
                    ].dropna()

                    if len(vals) > 0:
                        print(
                            f"{metric} | {pipeline}: "
                            f"mean={vals.mean():.3f}, std={vals.std():.3f}"
                        )
                    else:
                        print(f"{metric} | {pipeline}: no values")

            print(f"\nSaved:\n{out_csv}\n{wide_csv}")

    if all_results:
        all_combined = pd.concat(all_results, ignore_index=True)

        all_out_csv = OUT_DIR / f"motor_check_comparison_ALL_{ATLAS_TAG}.csv"
        all_combined.to_csv(all_out_csv, index=False)

        key_cols = ["dataset", "subject", "session", "task", "run", "atlas_tag"]

        all_wide = all_combined.pivot_table(
            index=key_cols,
            columns="pipeline",
            values=[
                "modularity_q",
                "motor_lr_corr",
                "fd_mean_filtered",
                "scrub_percent",
            ],
            aggfunc="first",
        )

        all_wide.columns = [f"{a}_{b}" for a, b in all_wide.columns]
        all_wide = all_wide.reset_index()

        if (
            "modularity_q_anatomical" in all_wide.columns
            and "modularity_q_global" in all_wide.columns
        ):
            all_wide["delta_modularity_global_minus_anatomical"] = (
                all_wide["modularity_q_global"] - all_wide["modularity_q_anatomical"]
            )

        if (
            "motor_lr_corr_anatomical" in all_wide.columns
            and "motor_lr_corr_global" in all_wide.columns
        ):
            all_wide["delta_motor_global_minus_anatomical"] = (
                all_wide["motor_lr_corr_global"] - all_wide["motor_lr_corr_anatomical"]
            )

        all_wide_csv = OUT_DIR / f"motor_check_comparison_ALL_{ATLAS_TAG}_wide.csv"
        all_wide.to_csv(all_wide_csv, index=False)

        print("\n===== OVERALL SUMMARY =====")

        for dataset_name in DATASETS_TO_RUN:
            print(f"\nDataset: {dataset_name}")

            subset = all_combined[all_combined["dataset"] == dataset_name]

            for metric in ["modularity_q", "motor_lr_corr"]:
                for pipeline in PIPELINES_TO_RUN:
                    vals = subset.loc[
                        subset["pipeline"] == pipeline,
                        metric,
                    ].dropna()

                    if len(vals) > 0:
                        print(
                            f"{metric} | {pipeline}: "
                            f"mean={vals.mean():.3f}, std={vals.std():.3f}"
                        )
                    else:
                        print(f"{metric} | {pipeline}: no values")

        print(f"\nSaved combined results:\n{all_out_csv}\n{all_wide_csv}")


if __name__ == "__main__":
    main()