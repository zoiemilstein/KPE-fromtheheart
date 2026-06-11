from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

import kpe_config as cfg


# ============================================================
# USER SETTINGS
# ============================================================

ATLAS_TAG = "schaefer400"

# Choose:
# ["tau"]
# ["yale"]
# ["combined"]
# ["tau", "yale", "combined"]
DATASETS_TO_RUN = ["tau", "yale", "combined"]

# Choose:
# ["anatomical"]
# ["global"]
# ["anatomical", "global"]
PIPELINES_TO_RUN = ["anatomical", "global"]

SESSION_COMPARISONS = [
    ("ses-1", "ses-2"),
    ("ses-1", "ses-3"),
]

# If True:
#   includes Vis__Vis, SomMot__SomMot, Default__Default, etc.
# If False:
#   only includes between-network pairs, like Limbic__Default.
INCLUDE_WITHIN_NETWORK = True


# ============================================================
# RESULT FILTERING / QUIET MODE
# ============================================================

# Main threshold for "worth looking at"
P_VALUE_THRESHOLD = 0.10

# Keep Cohen's d in the file, but do not filter by it for now.
# Later you can change this to 0.30 if you want.
MIN_ABS_COHENS_D = None

# Usually False. Turn True only when you want figures.
MAKE_PLOTS = False

# Saves an extra CSV with only rows that pass the threshold.
SAVE_INTERESTING_ONLY_FILE = True

# Print top 5 lowest p-values even if nothing passes threshold.
PRINT_TOP_IF_NOTHING_PASSES = True

# Optional biological filter.
# Keep False for now if you want to scan all network pairs.
FILTER_TO_PRIORITY_PAIRS_ONLY = False

PRIORITY_NETWORK_PAIRS = {
    "Limbic__Default",
    "SalVentAttn__Default",
    "SalVentAttn__Limbic",
    "Cont__Default",
    "DorsAttn__Default",
    "SomMot__Cont",
}


# ============================================================
# GROUP DEFINITIONS
# ============================================================

GROUP_1_TREATMENTS = ("ket0.5")
GROUP_2_TREATMENTS = ("placebo",)

GROUP_1_LABEL = "ketamine"
GROUP_2_LABEL = "placebo"

TOP_N_TO_PLOT = 10

OUT_DIR = cfg.RESULTS_DIR / "schaefer_connectivity_results_parcel_edge_average"
cfg.ensure_dir(OUT_DIR)


NETWORKS = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_subject_id(value) -> str:
    value = str(value).strip()

    if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
        digits = re.findall(r"\d+", value)
        return f"sub-{int(digits[0]):03d}" if digits else value

    match = re.match(r".*?(\d+)", value)
    return f"sub-{int(match.group(1)):03d}" if match else value


def normalize_session_label(session) -> str:
    session = str(session)

    m = re.search(r"(MRI|S|ses-)?(\d+)", session, flags=re.IGNORECASE)

    if m:
        return f"ses-{int(m.group(2))}"

    return session


def normalize_group_label(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def treatment_to_name(value) -> str | None:
    value = normalize_group_label(value)

    # TAU A/B/C coding
    if value == "a":
        return "ket0.5"
    if value == "b":
        return "ket0.2"
    if value == "c":
        return "placebo"

    # Explicit treatment names
    if value in ["ket0.5", "ketamine0.5", "ketamine05", "ket05"]:
        return "ket0.5"

    if value in ["ket0.2", "ketamine0.2", "ketamine02", "ket02"]:
        return "ket0.2"

    if value in ["placebo", "midazolam", "control"]:
        return "placebo"

    return None


def safe_name(text: str) -> str:
    text = str(text)
    text = text.replace(" ", "_")
    text = text.replace(".", "p")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(",", "_")

    return text


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def inverse_fisher_z(z):
    return np.tanh(z)


def fdr_bh(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """

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


def load_ts(path: Path):
    df = pd.read_csv(path)

    if "Background" in df.columns:
        df = df.drop(columns=["Background"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    return df


def get_network_from_column(col: str):
    col = str(col)

    for net in NETWORKS:
        if f"_{net}_" in col:
            return net

    return None


def cohen_d_independent(group_x, group_y):
    x = np.asarray(group_x, dtype=float)
    y = np.asarray(group_y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n_x, n_y = len(x), len(y)

    if n_x < 2 or n_y < 2:
        return np.nan

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    )

    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def label_interesting_results(results_df: pd.DataFrame):
    """
    Adds flags for which results are worth looking at.

    Main filter:
        p_value <= P_VALUE_THRESHOLD

    Cohen's d is kept in the output but not used as a filter unless
    MIN_ABS_COHENS_D is changed from None to a number.
    """

    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()

    df["abs_cohens_d"] = df["cohens_d"].abs()
    df["passes_p_threshold"] = df["p_value"] <= P_VALUE_THRESHOLD

    if MIN_ABS_COHENS_D is None:
        df["passes_effect_size_threshold"] = True
    else:
        df["passes_effect_size_threshold"] = df["abs_cohens_d"] >= MIN_ABS_COHENS_D

    if FILTER_TO_PRIORITY_PAIRS_ONLY:
        df["is_priority_pair"] = df["network_pair"].isin(PRIORITY_NETWORK_PAIRS)
    else:
        df["is_priority_pair"] = True

    df["is_interesting"] = (
        df["passes_p_threshold"]
        & df["passes_effect_size_threshold"]
        & df["is_priority_pair"]
    )

    df = df.sort_values(
        ["is_interesting", "p_value"],
        ascending=[False, True],
    )

    return df


# ============================================================
# RANDOMIZATION
# ============================================================

def read_randomization_file(randomization_path: Path) -> pd.DataFrame:
    if randomization_path.suffix.lower() == ".csv":
        return pd.read_csv(randomization_path)

    df = pd.read_excel(randomization_path)

    expected_any = {"SubID", "Group_Simbol", "scr_id", "Group"}

    if expected_any.intersection(set(df.columns)):
        return df

    df_header_1 = pd.read_excel(randomization_path, header=1)

    return df_header_1


def load_group_table(randomization_path: Path, dataset_name: str):
    if randomization_path is None:
        raise FileNotFoundError("No randomization file was found.")

    if not randomization_path.exists():
        raise FileNotFoundError(f"Randomization file not found: {randomization_path}")

    df = read_randomization_file(randomization_path)

    # -------- TAU table --------
    if "SubID" in df.columns and "Group_Simbol" in df.columns:
        subject_col = "SubID"
        group_col = "Group_Simbol"

    # -------- Yale table --------
    elif "scr_id" in df.columns and "Group" in df.columns:
        subject_col = "scr_id"
        group_col = "Group"

        if "Site" in df.columns:
            df = df[
                df["Site"].astype(str).str.lower() == dataset_name.lower()
            ].copy()

    else:
        raise KeyError(
            f"Unknown randomization table format.\n"
            f"Available columns are: {list(df.columns)}\n\n"
            f"Expected either:\n"
            f"TAU: SubID + Group_Simbol\n"
            f"Yale: scr_id + Group"
        )

    df = df[[subject_col, group_col]].dropna().drop_duplicates()

    subject_to_group = {}
    subject_to_treatment = {}

    for _, row in df.iterrows():
        subject = normalize_subject_id(row[subject_col])
        treatment = treatment_to_name(row[group_col])

        subject_to_treatment[subject] = treatment

        if treatment in GROUP_1_TREATMENTS:
            subject_to_group[subject] = "group1"

        elif treatment in GROUP_2_TREATMENTS:
            subject_to_group[subject] = "group2"

        else:
            subject_to_group[subject] = None

    print("\nRandomization loaded:")
    print(f"Dataset: {dataset_name}")
    print(f"Randomization file: {randomization_path}")
    print(f"Subjects in randomization: {len(subject_to_group)}")

    print("\nTreatment counts:")
    print(pd.Series(subject_to_treatment).value_counts(dropna=False))

    print("\nComparison group counts:")
    print(pd.Series(subject_to_group).value_counts(dropna=False))

    return subject_to_group, subject_to_treatment


# ============================================================
# SCHAEFER PARCEL-EDGE NETWORK CONNECTIVITY
# ============================================================

def compute_parcel_edge_network_connectivity(ts_df: pd.DataFrame):
    """
    Main method:
        1. compute parcel × parcel correlation matrix
        2. Fisher-z transform every parcel-level edge
        3. average all parcel-level edges inside each network pair/block
    """

    col_networks = {}
    valid_cols = []

    for col in ts_df.columns:
        net = get_network_from_column(col)

        if net is not None:
            col_networks[col] = net
            valid_cols.append(col)

    if len(valid_cols) < 300:
        raise ValueError(
            f"Too few Schaefer network columns found: {len(valid_cols)}"
        )

    ts_use = ts_df[valid_cols].copy()

    # Remove zero-variance columns, if any
    stds = ts_use.std(axis=0, skipna=True)
    good_cols = stds[stds > 0].index.tolist()

    ts_use = ts_use[good_cols]
    valid_cols = good_cols

    if ts_use.shape[1] < 300:
        raise ValueError(
            f"Too few valid Schaefer columns after zero-variance removal: {ts_use.shape[1]}"
        )

    corr = np.corrcoef(ts_use.values, rowvar=False)

    # Convert impossible/infinite values to nan
    corr[~np.isfinite(corr)] = np.nan

    zmat = fisher_z(corr)

    # Do not use diagonal self-correlations
    np.fill_diagonal(zmat, np.nan)

    parcel_networks = [col_networks[c] for c in valid_cols]

    network_counts = {
        net: int(sum(n == net for n in parcel_networks))
        for net in NETWORKS
    }

    pair_results = {}

    for i, net1 in enumerate(NETWORKS):

        if INCLUDE_WITHIN_NETWORK:
            start_j = i
        else:
            start_j = i + 1

        for net2 in NETWORKS[start_j:]:

            idx1 = [
                idx for idx, n in enumerate(parcel_networks)
                if n == net1
            ]

            idx2 = [
                idx for idx, n in enumerate(parcel_networks)
                if n == net2
            ]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            if net1 == net2:
                block = zmat[np.ix_(idx1, idx1)]
                iu = np.triu_indices_from(block, k=1)
                vals = block[iu]

            else:
                block = zmat[np.ix_(idx1, idx2)]
                vals = block.ravel()

            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                mean_z = np.nan
                mean_r = np.nan
            else:
                mean_z = float(np.mean(vals))
                mean_r = float(inverse_fisher_z(mean_z))

            pair = f"{net1}__{net2}"

            pair_results[pair] = {
                "network_1": net1,
                "network_2": net2,
                "z": mean_z,
                "r": mean_r,
                "n_edges": int(len(vals)),
            }

    return pair_results, network_counts


def extract_network_connectivity_for_dataset(dataset_name: str, pipeline_name: str):
    ts_dir = cfg.DATASETS[dataset_name][pipeline_name]
    randomization_path = cfg.DATASETS[dataset_name]["randomization"]

    print("\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Method: parcel-level FC matrix -> network-pair edge averaging")
    print(f"Time-series folder: {ts_dir}")
    print(f"Randomization file: {randomization_path}")
    print("==============================")

    if not ts_dir.exists():
        raise FileNotFoundError(f"Time-series folder not found: {ts_dir}")

    files = cfg.find_ts_files(ts_dir, ATLAS_TAG)

    subject_to_group, subject_to_treatment = load_group_table(
        randomization_path=randomization_path,
        dataset_name=dataset_name,
    )

    long_rows = []
    wide_rows = []
    skipped_files = []
    missing_randomization = []

    for f in files:
        try:
            ts = load_ts(f)

            if ts.shape[1] < 300:
                skipped_files.append((f.name, ts.shape[1], "too few columns"))
                continue

            pair_results, network_counts = compute_parcel_edge_network_connectivity(ts)

            meta = cfg.parse_entities(f.name)

            subject = normalize_subject_id(meta.get("subject"))
            session = normalize_session_label(meta.get("session"))

            group = subject_to_group.get(subject)
            treatment = subject_to_treatment.get(subject)

            if subject not in subject_to_group:
                missing_randomization.append(subject)

            subject_unique = f"{dataset_name}_{subject}"

            wide_row = {
                "dataset": dataset_name,
                "pipeline": pipeline_name,
                "subject": subject,
                "subject_unique": subject_unique,
                "session": session,
                "group": group,
                "treatment": treatment,
                "file": f.name,
                "n_timepoints": ts.shape[0],
                "n_parcels_input": ts.shape[1],
                "method": "parcel_fc_then_network_edge_average",
            }

            for net, count in network_counts.items():
                wide_row[f"n_parcels_{net}"] = count

            for pair, res in pair_results.items():
                net1 = res["network_1"]
                net2 = res["network_2"]
                r = res["r"]
                z = res["z"]
                n_edges = res["n_edges"]

                long_rows.append({
                    "dataset": dataset_name,
                    "pipeline": pipeline_name,
                    "subject": subject,
                    "subject_unique": subject_unique,
                    "session": session,
                    "group": group,
                    "treatment": treatment,
                    "network_1": net1,
                    "network_2": net2,
                    "network_pair": pair,
                    "r": r,
                    "z": z,
                    "n_edges_averaged": n_edges,
                    "file": f.name,
                    "method": "parcel_fc_then_network_edge_average",
                })

                wide_row[f"{pair}_r"] = r
                wide_row[f"{pair}_z"] = z
                wide_row[f"{pair}_n_edges"] = n_edges

            wide_rows.append(wide_row)

        except Exception as e:
            skipped_files.append((f.name, np.nan, str(e)))
            print(f"ERROR in {f.name}: {e}")

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    print("\n===== FILE QUALITY =====")
    print(f"Good scans: {len(wide_df)}")
    print(f"Skipped bad files: {len(skipped_files)}")

    if skipped_files:
        print("\nSkipped files:")
        for name, n_cols, reason in skipped_files[:30]:
            print(f"  {name} -> {reason} | columns={n_cols}")

    if missing_randomization:
        missing_unique = sorted(set(missing_randomization))
        print("\nWARNING: subjects missing from randomization:")
        print(missing_unique)

    if not wide_df.empty:
        print("\n===== SUBJECT COUNT =====")
        print("Subjects:", wide_df["subject"].nunique())
        print("Scans:", len(wide_df))
        print("Sessions:", sorted(wide_df["session"].dropna().unique()))

        print("\nTreatment counts in extracted data:")
        print(
            wide_df[["subject", "treatment"]]
            .drop_duplicates()["treatment"]
            .value_counts(dropna=False)
        )

        print("\nComparison group counts in extracted data:")
        print(
            wide_df[["subject", "group"]]
            .drop_duplicates()["group"]
            .value_counts(dropna=False)
        )

    return long_df, wide_df


# ============================================================
# GROUP COMPARISON
# ============================================================

def compute_between_group_delta_tests(
    wide_df: pd.DataFrame,
    dataset_label: str,
    pipeline_name: str,
    baseline_session: str,
    followup_session: str,
):
    baseline_session = normalize_session_label(baseline_session)
    followup_session = normalize_session_label(followup_session)

    z_cols = [
        c for c in wide_df.columns
        if c.endswith("_z")
    ]

    baseline_df = wide_df[wide_df["session"] == baseline_session].copy()
    followup_df = wide_df[wide_df["session"] == followup_session].copy()

    merged = baseline_df.merge(
        followup_df,
        on=["subject_unique", "group", "treatment", "dataset", "pipeline"],
        suffixes=("_baseline", "_followup"),
    )

    print("\n------------------------------")
    print(f"Group comparison: {dataset_label} | {pipeline_name}")
    print(f"{baseline_session} -> {followup_session}")
    print(f"Subjects with both sessions: {len(merged)}")
    print("------------------------------")

    print("\nSubjects by comparison group:")
    print(
        merged[["subject_unique", "group"]]
        .drop_duplicates()["group"]
        .value_counts(dropna=False)
    )

    rows = []

    for col in z_cols:
        base_col = f"{col}_baseline"
        follow_col = f"{col}_followup"

        if base_col not in merged.columns or follow_col not in merged.columns:
            continue

        network_pair = col.replace("_z", "")
        delta_col = f"{network_pair}_delta"

        merged[delta_col] = merged[follow_col] - merged[base_col]

        group1_delta = merged.loc[
            merged["group"] == "group1",
            delta_col,
        ].dropna()

        group2_delta = merged.loc[
            merged["group"] == "group2",
            delta_col,
        ].dropna()

        if len(group1_delta) >= 2 and len(group2_delta) >= 2:
            t_stat, p_value = ttest_ind(
                group1_delta,
                group2_delta,
                equal_var=False,
            )

            rows.append({
                "dataset_analysis": dataset_label,
                "pipeline": pipeline_name,
                "baseline_session": baseline_session,
                "followup_session": followup_session,
                "comparison": f"{GROUP_1_LABEL} vs {GROUP_2_LABEL}",
                "group_1_treatments": ",".join(GROUP_1_TREATMENTS),
                "group_2_treatments": ",".join(GROUP_2_TREATMENTS),
                "network_pair": network_pair,
                "n_group1": len(group1_delta),
                "n_group2": len(group2_delta),
                "mean_delta_group1": group1_delta.mean(),
                "mean_delta_group2": group2_delta.mean(),
                "mean_diff_group1_minus_group2": group1_delta.mean() - group2_delta.mean(),
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": cohen_d_independent(group1_delta, group2_delta),
            })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df["p_fdr_bh"] = fdr_bh(results_df["p_value"].to_numpy())
        results_df["p_bonferroni"] = np.minimum(
            results_df["p_value"] * len(results_df),
            1.0,
        )
        results_df = results_df.sort_values("p_value")

    return results_df, merged


# ============================================================
# OUTPUT / PRINTING
# ============================================================

def print_interesting_summary(results_df: pd.DataFrame, interesting_df: pd.DataFrame, label: str):
    print("\n===== INTERESTING RESULTS =====")
    print(label)

    cols_to_print = [
        "network_pair",
        "n_group1",
        "n_group2",
        "mean_delta_group1",
        "mean_delta_group2",
        "mean_diff_group1_minus_group2",
        "p_value",
        "p_fdr_bh",
        "cohens_d",
        "abs_cohens_d",
    ]

    if results_df is None or results_df.empty:
        print("No statistical results were computed.")
        return

    available_cols = [c for c in cols_to_print if c in results_df.columns]

    if interesting_df.empty:
        print(f"No results passed p <= {P_VALUE_THRESHOLD}")

        if PRINT_TOP_IF_NOTHING_PASSES:
            print("\nLowest p-values anyway:")
            print(
                results_df[available_cols]
                .sort_values("p_value")
                .head(5)
                .to_string(index=False)
            )

    else:
        print(f"Rows passing p <= {P_VALUE_THRESHOLD}: {len(interesting_df)}")
        print(
            interesting_df[available_cols]
            .sort_values("p_value")
            .to_string(index=False)
        )


def plot_top_network_results(results_df: pd.DataFrame, output_path: Path, title: str):
    if results_df is None or results_df.empty:
        print("No results to plot.")
        return

    top_df = results_df.sort_values("p_value").head(TOP_N_TO_PLOT).copy()
    top_df = top_df.sort_values("mean_diff_group1_minus_group2")

    plt.figure(figsize=(11, 7))

    colors = [
        "red" if x > 0 else "blue"
        for x in top_df["mean_diff_group1_minus_group2"]
    ]

    plt.barh(
        top_df["network_pair"],
        top_df["mean_diff_group1_minus_group2"],
        color=colors,
        alpha=0.75,
    )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel(f"Mean Δ difference: {GROUP_1_LABEL} − {GROUP_2_LABEL}")
    plt.ylabel("Network pair")
    plt.title(title)

    for i, (_, row) in enumerate(top_df.iterrows()):
        label = f"  p={row['p_value']:.3f}"

        if "p_fdr_bh" in row and pd.notna(row["p_fdr_bh"]):
            label += f", q={row['p_fdr_bh']:.3f}"

        plt.text(
            row["mean_diff_group1_minus_group2"],
            i,
            label,
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    all_long = []
    all_wide = []
    all_results = []
    all_interesting = []

    datasets_needed = set()

    for d in DATASETS_TO_RUN:
        if d == "combined":
            datasets_needed.update(["tau", "yale"])
        else:
            datasets_needed.add(d)

    print("\nDatasets requested:", DATASETS_TO_RUN)
    print("Datasets to extract:", sorted(datasets_needed))
    print("Pipelines to run:", PIPELINES_TO_RUN)
    print("Include within-network pairs:", INCLUDE_WITHIN_NETWORK)
    print("P-value threshold for interesting results:", P_VALUE_THRESHOLD)
    print("Filtering by Cohen's d:", MIN_ABS_COHENS_D)
    print("Make plots:", MAKE_PLOTS)
    print(
        f"Comparison: {GROUP_1_LABEL} {GROUP_1_TREATMENTS} "
        f"vs {GROUP_2_LABEL} {GROUP_2_TREATMENTS}"
    )

    # 1. Extract connectivity
    for pipeline_name in PIPELINES_TO_RUN:
        for dataset_name in sorted(datasets_needed):

            long_df, wide_df = extract_network_connectivity_for_dataset(
                dataset_name=dataset_name,
                pipeline_name=pipeline_name,
            )

            long_out = OUT_DIR / (
                f"{dataset_name}_{pipeline_name}_"
                f"schaefer_parcel_edge_network_long.csv"
            )

            wide_out = OUT_DIR / (
                f"{dataset_name}_{pipeline_name}_"
                f"schaefer_parcel_edge_network_wide.csv"
            )

            long_df.to_csv(long_out, index=False)
            wide_df.to_csv(wide_out, index=False)

            print(f"Saved extracted long data: {long_out}")
            print(f"Saved extracted wide data: {wide_out}")

            if not long_df.empty:
                all_long.append(long_df)

            if not wide_df.empty:
                all_wide.append(wide_df)

    if not all_wide:
        print("No data extracted.")
        return

    combined_long = pd.concat(all_long, ignore_index=True)
    combined_wide = pd.concat(all_wide, ignore_index=True)

    all_long_path = OUT_DIR / "ALL_extracted_schaefer_parcel_edge_network_long.csv"
    all_wide_path = OUT_DIR / "ALL_extracted_schaefer_parcel_edge_network_wide.csv"

    combined_long.to_csv(all_long_path, index=False)
    combined_wide.to_csv(all_wide_path, index=False)

    print(f"\nSaved all extracted long data: {all_long_path}")
    print(f"Saved all extracted wide data: {all_wide_path}")

    # 2. Run group analysis
    for pipeline_name in PIPELINES_TO_RUN:

        for dataset_label in DATASETS_TO_RUN:

            if dataset_label == "combined":
                analysis_df = combined_wide[
                    combined_wide["pipeline"] == pipeline_name
                ].copy()

            else:
                analysis_df = combined_wide[
                    (combined_wide["dataset"] == dataset_label)
                    & (combined_wide["pipeline"] == pipeline_name)
                ].copy()

            if analysis_df.empty:
                print(f"No data for {dataset_label} | {pipeline_name}")
                continue

            for baseline_session, followup_session in SESSION_COMPARISONS:

                results_df, subject_delta_df = compute_between_group_delta_tests(
                    wide_df=analysis_df,
                    dataset_label=dataset_label,
                    pipeline_name=pipeline_name,
                    baseline_session=baseline_session,
                    followup_session=followup_session,
                )

                results_df = label_interesting_results(results_df)

                base_clean = normalize_session_label(baseline_session)
                follow_clean = normalize_session_label(followup_session)

                comparison_name = (
                    f"{safe_name('_'.join(GROUP_1_TREATMENTS))}"
                    f"_vs_"
                    f"{safe_name('_'.join(GROUP_2_TREATMENTS))}"
                )

                within_tag = (
                    "with_within_network"
                    if INCLUDE_WITHIN_NETWORK
                    else "between_network_only"
                )

                base_file_name = (
                    f"{dataset_label}_{pipeline_name}_"
                    f"{base_clean}_to_{follow_clean}_"
                    f"{comparison_name}_"
                    f"{within_tag}"
                )

                results_path = OUT_DIR / f"{base_file_name}_network_results.csv"
                deltas_path = OUT_DIR / f"{base_file_name}_subject_deltas.csv"

                results_df.to_csv(results_path, index=False)
                subject_delta_df.to_csv(deltas_path, index=False)

                print(f"\nSaved all group results: {results_path}")
                print(f"Saved subject deltas: {deltas_path}")

                interesting_df = pd.DataFrame()

                if results_df is not None and not results_df.empty:
                    interesting_df = results_df[results_df["is_interesting"]].copy()

                    if SAVE_INTERESTING_ONLY_FILE:
                        interesting_path = OUT_DIR / f"{base_file_name}_INTERESTING_results.csv"
                        interesting_df.to_csv(interesting_path, index=False)
                        print(f"Saved interesting-only results: {interesting_path}")

                    label = f"{dataset_label} | {pipeline_name} | {base_clean} -> {follow_clean}"
                    print_interesting_summary(results_df, interesting_df, label)

                    all_results.append(results_df)

                    if not interesting_df.empty:
                        temp = interesting_df.copy()
                        temp["analysis_label"] = label
                        all_interesting.append(temp)

                    if MAKE_PLOTS and not interesting_df.empty:
                        fig_path = OUT_DIR / f"{base_file_name}_INTERESTING_top_network_results.png"

                        title = (
                            f"Interesting Schaefer parcel-edge network changes\n"
                            f"{dataset_label} | {pipeline_name} | "
                            f"{base_clean} → {follow_clean}\n"
                            f"{GROUP_1_LABEL}: {GROUP_1_TREATMENTS} "
                            f"vs {GROUP_2_LABEL}: {GROUP_2_TREATMENTS}"
                        )

                        plot_top_network_results(
                            results_df=interesting_df,
                            output_path=fig_path,
                            title=title,
                        )

    # 3. Save global summary files
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)

        comparison_name = (
            f"{safe_name('_'.join(GROUP_1_TREATMENTS))}"
            f"_vs_"
            f"{safe_name('_'.join(GROUP_2_TREATMENTS))}"
        )

        within_tag = (
            "with_within_network"
            if INCLUDE_WITHIN_NETWORK
            else "between_network_only"
        )

        all_results_path = OUT_DIR / (
            f"ALL_{comparison_name}_{within_tag}_network_results.csv"
        )

        all_results_df.to_csv(all_results_path, index=False)

        print(f"\nSaved all group results: {all_results_path}")

    if all_interesting:
        all_interesting_df = pd.concat(all_interesting, ignore_index=True)

        all_interesting_path = OUT_DIR / (
            f"ALL_INTERESTING_p_lt_{safe_name(str(P_VALUE_THRESHOLD))}_network_results.csv"
        )

        all_interesting_df.to_csv(all_interesting_path, index=False)

        print("\n===== ALL INTERESTING RESULTS ACROSS EVERYTHING =====")
        print(f"Total interesting rows: {len(all_interesting_df)}")
        print(f"Saved: {all_interesting_path}")

        cols = [
            "analysis_label",
            "network_pair",
            "n_group1",
            "n_group2",
            "mean_diff_group1_minus_group2",
            "p_value",
            "p_fdr_bh",
            "cohens_d",
        ]

        cols = [c for c in cols if c in all_interesting_df.columns]

        print(
            all_interesting_df[cols]
            .sort_values("p_value")
            .to_string(index=False)
        )

    else:
        print("\n===== ALL INTERESTING RESULTS ACROSS EVERYTHING =====")
        print(f"No result passed p <= {P_VALUE_THRESHOLD}.")

    print("\n=== DONE ===")
    print("All results saved under:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()