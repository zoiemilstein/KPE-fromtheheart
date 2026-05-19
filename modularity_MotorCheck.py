from pathlib import Path
import kpe_config as cfg
import numpy as np
import pandas as pd
import networkx as nx

NO_GSR_DIR = cfg.ANATOMICAL_TS_DIR
GSR_DIR = cfg.GLOBAL_TS_DIR

NO_GSR_SCRUB_CSV = cfg.ANATOMICAL_SCRUB_CSV
GSR_SCRUB_CSV = cfg.GLOBAL_SCRUB_CSV

OUT_DIR = cfg.RESULTS_DIR / "motor_check_results"
cfg.ensure_dir(OUT_DIR)

ATLAS_TAG = "schaefer400"   # "schaefer400"
ZERO_NEGATIVE_EDGES_FOR_MODULARITY = True


def get_timeseries_files(ts_dir: Path, atlas_tag: str):
    files = sorted(ts_dir.glob("*.csv"))
    files = [
        f for f in files
        if atlas_tag.lower() in f.name.lower()
        and "scrubbing_report" not in f.name.lower()
        and "motor_check_results" not in f.name.lower()
    ]
    return files


def parse_file_metadata(fname: str):
    out = {
        "file": fname,
        "subject": "",
        "session": "",
        "task": "",
        "run": "",
    }
    for key in ["sub", "ses", "task", "run"]:
        m = re.search(rf"({key}-[^_]+)", fname)
        if m:
            if key == "sub":
                out["subject"] = m.group(1)
            elif key == "ses":
                out["session"] = m.group(1)
            elif key == "task":
                out["task"] = m.group(1)
            elif key == "run":
                out["run"] = m.group(1)
    return out


def load_numeric_ts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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

    Why zero negative edges?
    Standard Louvain modularity is most straightforward for positive weights.
    After GSR, negative correlations become more common, and handling them
    in a simple, interpretable way is tricky. So here we keep only positive
    connectivity and ask: how strongly does the graph separate into positively
    connected communities?
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

    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    q = nx.community.modularity(G, communities, weight="weight")
    return float(q)


def find_schaefer_sommot_columns(columns):
    """
    Uses the actual Schaefer 7-network naming convention:
      7Networks_LH_SomMot_*
      7Networks_RH_SomMot_*
    """
    left_cols = [c for c in columns if "LH_SomMot" in c]
    right_cols = [c for c in columns if "RH_SomMot" in c]
    return left_cols, right_cols


def compute_motor_lr_schaefer(ts_df: pd.DataFrame):
    """
    Motor sanity check:
    1. take all left SomMot parcels
    2. average them at each timepoint -> one left SomMot signal
    3. take all right SomMot parcels
    4. average them at each timepoint -> one right SomMot signal
    5. correlate left vs right mean signal across time

    This is a signal-preservation check, not a graph metric.
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


def load_scrub_report(scrub_csv: Path):
    if not scrub_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(scrub_csv)

    keep_cols = [c for c in [
        "subject", "session", "task", "run",
        "fd_mean_raw", "fd_mean_filtered", "fd_max_filtered",
        "raw_spikes", "filtered_spikes",
        "scrubbed_volumes", "scrub_percent",
        "high_motion_skip", "gsr_applied"
    ] if c in df.columns]

    df = df[keep_cols].copy()
    df["task"] = "task-" + df["task"].astype(str).str.replace("task-", "", regex=False)

    return df


def build_metrics_for_dir(ts_dir: Path, scrub_csv: Path, atlas_tag: str, pipeline_name: str):
    files = cfg.find_ts_files(ts_dir, atlas_tag)
    rows = []

    for f in files:
        try:
            row = process_one_file(f, atlas_tag)
            row["pipeline"] = pipeline_name
            rows.append(row)

        except Exception as e:
            print(f"[{pipeline_name}] ERROR in {f.name}: {e}")

    metrics_df = pd.DataFrame(rows)
    scrub_df = load_scrub_report(scrub_csv)

    if len(metrics_df) == 0:
        return metrics_df

    if len(scrub_df) > 0:
        metrics_df = metrics_df.merge(
            scrub_df,
            on=["subject", "session", "task", "run"],
            how="left"
        )

    return metrics_df


def main():
    anatomical_df = build_metrics_for_dir(
        NO_GSR_DIR, NO_GSR_SCRUB_CSV, ATLAS_TAG, "anatomical"
    )

    global_df = build_metrics_for_dir(
        GSR_DIR, GSR_SCRUB_CSV, ATLAS_TAG, "global"
    )

    combined = pd.concat([anatomical_df, global_df], axis=0, ignore_index=True)

    print("\n===== SUMMARY =====")

    for metric in ["modularity_q", "motor_lr_corr"]:
        for pipeline in ["anatomical", "global"]:
            vals = combined.loc[combined["pipeline"] == pipeline, metric].dropna()
            print(f"{metric} | {pipeline}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    out_csv = OUT_DIR / f"motor_check_comparison_{ATLAS_TAG}.csv"
    combined.to_csv(out_csv, index=False)

    key_cols = ["subject", "session", "task", "run", "atlas_tag"]
    wide = combined.pivot_table(
        index=key_cols,
        columns="pipeline",
        values=["modularity_q", "motor_lr_corr", "fd_mean_filtered", "scrub_percent"],
        aggfunc="first"
    )

    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    if "modularity_q_anatomical" in wide.columns and "modularity_q_global" in wide.columns:
        wide["delta_modularity_global_minus_anatomical"] = (
            wide["modularity_q_global"] - wide["modularity_q_anatomical"]
        )

    if "motor_lr_corr_anatomical" in wide.columns and "motor_lr_corr_global" in wide.columns:
        wide["delta_motor_global_minus_anatomical"] = (
            wide["motor_lr_corr_global"] - wide["motor_lr_corr_anatomical"]
        )

    wide_csv = OUT_DIR / f"motor_check_comparison_{ATLAS_TAG}_wide.csv"
    wide.to_csv(wide_csv, index=False)

    print(f"\nSaved:\n{out_csv}\n{wide_csv}")


if __name__ == "__main__":
    main()