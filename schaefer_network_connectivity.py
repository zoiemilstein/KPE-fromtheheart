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

# Same session comparisons for TAU and Yale.
# The code converts MRI1 / S1 / ses-MRI1 / ses-1 -> ses-1
SESSION_COMPARISONS = [
    ("ses-1", "ses-2"),
    ("ses-1", "ses-3"),
]


# ============================================================
# GROUP DEFINITIONS
# ============================================================
# Treatment names are standardized to:
# ket0.5
# ket0.2
# placebo
#
# Choose comparison here:
# ("ket0.5", "ket0.2") vs ("placebo",) = all ketamine vs placebo
# ("ket0.5",) vs ("ket0.2",) = high dose vs low dose
# ("ket0.5",) vs ("placebo",) = high dose vs placebo
# ("ket0.2",) vs ("placebo",) = low dose vs placebo

GROUP_1_TREATMENTS = ("ket0.5", "ket0.2")
GROUP_2_TREATMENTS = ("placebo")

GROUP_1_LABEL = "ketamine"
GROUP_2_LABEL = "placebo"

TOP_N_TO_PLOT = 10

OUT_DIR = cfg.RESULTS_DIR / "schaefer_connectivity_results"
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
    """
    Converts:
    KPE1780 -> sub-1780
    sub-1780 -> sub-1780
    1780 -> sub-1780
    """

    value = str(value).strip()

    if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
        digits = re.findall(r"\d+", value)
        return f"sub-{int(digits[0]):03d}" if digits else value

    match = re.match(r".*?(\d+)", value)
    return f"sub-{int(match.group(1)):03d}" if match else value


def normalize_session_label(session) -> str:
    """
    Converts:
    MRI1, S1, ses-MRI1, ses-S1, ses-1
    into:
    ses-1
    """

    session = str(session)

    m = re.search(r"(MRI|S|ses-)?(\d+)", session, flags=re.IGNORECASE)

    if m:
        return f"ses-{int(m.group(2))}"

    return session


def normalize_group_label(value) -> str:
    """
    General text cleanup for treatment/group names.
    """

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def treatment_to_name(value) -> str | None:
    """
    Converts both TAU and Yale randomization values into:
    ket0.5 / ket0.2 / placebo

    TAU:
        A -> ket0.5
        B -> ket0.2
        C -> placebo

    Yale:
        Ket0.5 -> ket0.5
        Ket0.2 -> ket0.2
        Placebo -> placebo
    """

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
    """
    Makes a string safe for filenames.
    """

    text = str(text)
    text = text.replace(" ", "_")
    text = text.replace(".", "p")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(",", "_")
    return text


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def load_ts(path: Path):
    df = pd.read_csv(path)

    if "Background" in df.columns:
        df = df.drop(columns=["Background"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    return df


def get_network_from_column(col: str):
    for net in NETWORKS:
        if f"_{net}_" in col:
            return net

    return None


def make_network_timeseries(ts_df: pd.DataFrame):
    network_ts = {}

    for net in NETWORKS:
        cols = [
            c for c in ts_df.columns
            if get_network_from_column(c) == net
        ]

        if len(cols) == 0:
            print(f"WARNING: no columns found for network {net}")
            continue

        network_ts[net] = ts_df[cols].mean(axis=1)

    return pd.DataFrame(network_ts)


def cohen_d_independent(group_x, group_y):
    x = np.asarray(group_x)
    y = np.asarray(group_y)

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


# ============================================================
# RANDOMIZATION
# ============================================================

def read_randomization_file(randomization_path: Path) -> pd.DataFrame:
    """
    Reads CSV or Excel.
    If Excel has a weird first row, tries header=1 as fallback.
    """

    if randomization_path.suffix.lower() == ".csv":
        return pd.read_csv(randomization_path)

    df = pd.read_excel(randomization_path)

    expected_any = {"SubID", "Group_Simbol", "scr_id", "Group"}
    if expected_any.intersection(set(df.columns)):
        return df

    # Fallback for Yale files where the real header may be on row 2
    df_header_1 = pd.read_excel(randomization_path, header=1)
    return df_header_1


def load_group_table(randomization_path: Path, dataset_name: str):
    """
    Supports two randomization formats:

    TAU table:
        SubID | Group_Simbol
        A -> ket0.5
        B -> ket0.2
        C -> placebo

    Yale table:
        Site | scr_id | Group
        Ket0.5 -> ket0.5
        Ket0.2 -> ket0.2
        Placebo -> placebo

    Returns:
        subject_to_group:
            sub-XXXX -> group1 / group2 / None

        subject_to_treatment:
            sub-XXXX -> ket0.5 / ket0.2 / placebo / None
    """

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
# EXTRACT SCHAEFER NETWORK CONNECTIVITY
# ============================================================

def extract_network_connectivity_for_dataset(dataset_name: str, pipeline_name: str):
    ts_dir = cfg.DATASETS[dataset_name][pipeline_name]
    randomization_path = cfg.DATASETS[dataset_name]["randomization"]

    print("\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"Pipeline: {pipeline_name}")
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
                skipped_files.append((f.name, ts.shape[1]))
                continue

            net_ts = make_network_timeseries(ts)

            if net_ts.shape[1] != 7:
                print(f"WARNING: {f.name} produced only {net_ts.shape[1]} networks")
                continue

            corr = net_ts.corr()
            meta = cfg.parse_entities(f.name)

            subject = normalize_subject_id(meta.get("subject"))
            session = normalize_session_label(meta.get("session"))

            group = subject_to_group.get(subject)
            treatment = subject_to_treatment.get(subject)

            if subject not in subject_to_group:
                missing_randomization.append(subject)

            # Important for combined analysis:
            # prevents tau sub-001 and yale sub-001 from being treated as same person
            subject_unique = f"{dataset_name}_{subject}"

            wide_row = {
                "dataset": dataset_name,
                "pipeline": pipeline_name,
                "subject": subject,
                "subject_unique": subject_unique,
                "session": session,
                "group": group,             # group1 / group2 / None
                "treatment": treatment,     # ket0.5 / ket0.2 / placebo / None
                "file": f.name,
                "n_timepoints": ts.shape[0],
                "n_parcels": ts.shape[1],
            }

            for i, net1 in enumerate(NETWORKS):
                for net2 in NETWORKS[i + 1:]:

                    if net1 not in corr.index or net2 not in corr.columns:
                        continue

                    r = corr.loc[net1, net2]
                    z = fisher_z(r)
                    pair = f"{net1}__{net2}"

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
                        "file": f.name,
                    })

                    wide_row[f"{pair}_r"] = r
                    wide_row[f"{pair}_z"] = z

            wide_rows.append(wide_row)

        except Exception as e:
            print(f"ERROR in {f.name}: {e}")

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    print("\n===== FILE QUALITY =====")
    print(f"Good scans: {len(wide_df)}")
    print(f"Skipped bad files: {len(skipped_files)}")

    if skipped_files:
        print("\nSkipped files:")
        for name, n_cols in skipped_files:
            print(f"  {name} -> got {n_cols} columns")

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

    z_cols = [c for c in wide_df.columns if c.endswith("_z")]

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
            delta_col
        ].dropna()

        group2_delta = merged.loc[
            merged["group"] == "group2",
            delta_col
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
        results_df = results_df.sort_values("p_value")

    return results_df, merged


# ============================================================
# PLOTTING
# ============================================================

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
        plt.text(
            row["mean_diff_group1_minus_group2"],
            i,
            f"  p={row['p_value']:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    all_long = []
    all_wide = []

    # If user asks for combined, extract both tau and yale
    datasets_needed = set()

    for d in DATASETS_TO_RUN:
        if d == "combined":
            datasets_needed.update(["tau", "yale"])
        else:
            datasets_needed.add(d)

    print("\nDatasets requested:", DATASETS_TO_RUN)
    print("Datasets to extract:", sorted(datasets_needed))
    print("Pipelines to run:", PIPELINES_TO_RUN)
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

            long_out = OUT_DIR / f"{dataset_name}_{pipeline_name}_schaefer_network_long.csv"
            wide_out = OUT_DIR / f"{dataset_name}_{pipeline_name}_schaefer_network_wide.csv"

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

    all_long_path = OUT_DIR / "ALL_extracted_schaefer_network_long.csv"
    all_wide_path = OUT_DIR / "ALL_extracted_schaefer_network_wide.csv"

    combined_long.to_csv(all_long_path, index=False)
    combined_wide.to_csv(all_wide_path, index=False)

    print(f"\nSaved all extracted long data: {all_long_path}")
    print(f"Saved all extracted wide data: {all_wide_path}")

    all_results = []

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

                base_clean = normalize_session_label(baseline_session)
                follow_clean = normalize_session_label(followup_session)

                comparison_name = (
                    f"{safe_name('_'.join(GROUP_1_TREATMENTS))}"
                    f"_vs_"
                    f"{safe_name('_'.join(GROUP_2_TREATMENTS))}"
                )

                results_path = OUT_DIR / (
                    f"{dataset_label}_{pipeline_name}_"
                    f"{base_clean}_to_{follow_clean}_"
                    f"{comparison_name}_network_results.csv"
                )

                deltas_path = OUT_DIR / (
                    f"{dataset_label}_{pipeline_name}_"
                    f"{base_clean}_to_{follow_clean}_"
                    f"{comparison_name}_subject_deltas.csv"
                )

                results_df.to_csv(results_path, index=False)
                subject_delta_df.to_csv(deltas_path, index=False)

                print(f"Saved group results: {results_path}")
                print(f"Saved subject deltas: {deltas_path}")

                if not results_df.empty:
                    all_results.append(results_df)

                    fig_path = OUT_DIR / (
                        f"{dataset_label}_{pipeline_name}_"
                        f"{base_clean}_to_{follow_clean}_"
                        f"{comparison_name}_top_network_results.png"
                    )

                    title = (
                        f"Top Schaefer network connectivity changes\n"
                        f"{dataset_label} | {pipeline_name} | "
                        f"{base_clean} → {follow_clean}\n"
                        f"{GROUP_1_LABEL}: {GROUP_1_TREATMENTS} "
                        f"vs {GROUP_2_LABEL}: {GROUP_2_TREATMENTS}"
                    )

                    plot_top_network_results(
                        results_df=results_df,
                        output_path=fig_path,
                        title=title,
                    )

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)

        comparison_name = (
            f"{safe_name('_'.join(GROUP_1_TREATMENTS))}"
            f"_vs_"
            f"{safe_name('_'.join(GROUP_2_TREATMENTS))}"
        )

        all_results_path = OUT_DIR / f"ALL_{comparison_name}_network_results.csv"
        all_results_df.to_csv(all_results_path, index=False)

        print(f"\nSaved all group results: {all_results_path}")

    print("\n=== DONE ===")
    print("All results saved under:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()