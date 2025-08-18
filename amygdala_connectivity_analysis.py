import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import ttest_rel, ttest_ind
from nilearn import datasets, plotting
from nilearn.plotting import plot_matrix

# =============================================================================
# CONFIG
# =============================================================================
# It yields, yet remains whole Roots in the earth, spirit in the sky
scrubbed_volumes_threshold = 115

PROJECT_ROOT = r"C:\aaf-files"  # Folder containing *_aal_ts.csv time series
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "t_test_results")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Show a matrix figure for each correlation matrix (off by default, as it can be many)
SHOW_CORRELATION_MATRICES = False

# Group table location (update if needed)
RANDOMIZATION_XLSX_PATH = "C:/Users/amirh/Downloads/RandomizationTable.xlsx"
report_path = "C:/AALpath/sub-987_ses-MRI1_aal_ts/scrubbing_report.csv"

# Which sessions to compare? (substring matching, case-insensitive)
# e.g., "MRI1" or "S1" for baseline; "MRI2" or "S2" for follow-up
BASELINE_SESSION_KEYWORDS = ("MRI1", "S1")
FOLLOWUP_SESSION_KEYWORDS = ("MRI2", "S2")

# Column names inside the randomization Excel
RANDOMIZATION_SUBJECT_COLUMN = "SubID"
RANDOMIZATION_GROUP_COLUMN = "Group_Simbol"
KETAMINE_GROUP_SYMBOLS = ("A",)
CONTROL_GROUP_SYMBOLS = ("C",)

# === NEW: manual control for group analysis (with auto-fallback if groups missing) ===
ANALYZE_BY_GROUP = True
# =============================================================================


# =============================================================================
# Utilities
# =============================================================================
def filter_correlation_matrices_by_fd_motion_threshold(report_path, correlation_matrices):
    scrub_report = pd.read_csv(report_path)
    over_scrubbed_volumes = scrub_report[scrub_report["scrubbed_volumes"] > scrubbed_volumes_threshold]
    subject_and_session_to_delete = set(zip(over_scrubbed_volumes["subject"], over_scrubbed_volumes["session"]))

    def parse_subject_session(key):
        base = os.path.basename(key).replace("_aal_ts.csv", "")
        parts = base.split("_")
        return parts[0], parts[1]  # ('sub-010', 'ses-MRI1')

    deleted_str = "none" if not subject_and_session_to_delete else ", ".join(
        f"{sub}_{ses}" for sub, ses in sorted(subject_and_session_to_delete)
    )
    print(f"Deleted subjects and session: {deleted_str}")

    return {
        key: mat
        for key, mat in correlation_matrices.items()
        if (parse_subject_session(key) not in subject_and_session_to_delete)
    }


def load_group_table(
        xlsx_path: str,
        subject_col: str = "SubID",
        group_col: str = "Group_Simbol",
        ketamine_groups=("A", "B"),
        control_groups=("C",),
):
    data_frame = pd.read_excel(xlsx_path)

    def normalize_subject_id(value: str) -> str:
        value = str(value).strip()
        if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
            # Ensure zero-padded to 3 digits if already in 'sub-###' form
            digits = re.findall(r"\d+", value)
            return f"sub-{int(digits[0]):03d}" if digits else value
        match = re.match(r".*?(\d+)", value)
        return f"sub-{int(match.group(1)):03d}" if match else value

    data_frame["_subject_norm"] = data_frame[subject_col].apply(normalize_subject_id)
    data_frame["_group_norm"] = data_frame[group_col].astype(str).str.strip().str.upper()

    subject_to_group = {}
    for _, row in data_frame.iterrows():
        group_symbol = row["_group_norm"]
        if group_symbol in ketamine_groups:
            subject_to_group[row["_subject_norm"]] = "ketamine"
        elif group_symbol in control_groups:
            subject_to_group[row["_subject_norm"]] = "control"
        else:
            subject_to_group[row["_subject_norm"]] = None

    return subject_to_group, data_frame


def resolve_session_labels(
        amygdala_correlations: dict,
        baseline_keywords: tuple,
        followup_keywords: tuple,
) -> tuple:
    """
    Resolve which session labels in amygdala_correlations correspond to baseline and follow-up,
    using substring matching with the provided keyword tuples.

    Returns:
        (baseline_session_label, followup_session_label)
    Raises:
        ValueError if one or both sessions cannot be found.
    """
    all_session_labels = set(session for (_, session) in amygdala_correlations.keys())

    def pick_label(keyword_tuple):
        keyword_tuple = tuple(k.lower() for k in keyword_tuple)
        for label in all_session_labels:
            lower_label = label.lower()
            if any(k in lower_label for k in keyword_tuple):
                return label
        return None

    baseline_label = pick_label(baseline_keywords)
    followup_label = pick_label(followup_keywords)

    if not baseline_label or not followup_label:
        raise ValueError(
            f"Could not resolve sessions. Found labels: {sorted(all_session_labels)}. "
            f"Baseline keywords: {baseline_keywords}, Follow-up keywords: {followup_keywords}"
        )
    return baseline_label, followup_label


# =============================================================================
# Data I/O and correlation extraction
# =============================================================================
def compute_pearson_correlations(project_root: str) -> dict:
    """
    Create correlation matrices (Pandas DataFrame) from each *_aal_ts.csv found in project_root.

    Returns:
        dict[str, pd.DataFrame]: mapping from filename -> correlation matrix
    """
    correlation_matrices = {}

    for ts_file in os.listdir(project_root):
        if not ts_file.endswith(".csv"):
            continue
        file_path = os.path.join(project_root, ts_file)
        try:
            data_frame = pd.read_csv(file_path)
            if "Amygdala_L" in data_frame.columns and "Amygdala_R" in data_frame.columns:
                correlation_matrix = data_frame.corr()
                correlation_matrices[ts_file] = correlation_matrix
                print(f"Created correlation matrix for {ts_file}")
            else:
                print(f"Warning: {ts_file} missing amygdala columns (Amygdala_L / Amygdala_R)")
        except Exception as error:
            print(f"Error processing {ts_file}: {error}")

    if SHOW_CORRELATION_MATRICES:
        for filename, corr_mat in correlation_matrices.items():
            plot_matrix(corr_mat, vmax=0.8, vmin=-0.8, colorbar=True)
            plt.title(filename, fontsize=10)
            plt.tight_layout()
            plt.show()

    return correlation_matrices


def extract_amygdala_correlations(correlation_matrices: dict) -> dict:
    amygdala_correlations = {}

    for file_name, corr_df in correlation_matrices.items():
        left_amygdala_corr = corr_df.loc["Amygdala_L", :].drop("Amygdala_L")
        right_amygdala_corr = corr_df.loc["Amygdala_R", :].drop("Amygdala_R")

        # Expect something like: sub-024_ses-MRI1_aal_ts.csv
        name_without_suffix = file_name.replace("_aal_ts.csv", "")
        name_parts = name_without_suffix.split("_")
        subject_id = name_parts[0]  # e.g., 'sub-024'
        session_label = name_parts[1]  # e.g., 'ses-MRI1'

        amygdala_correlations[(subject_id, session_label)] = {
            "Amygdala_L": left_amygdala_corr,
            "Amygdala_R": right_amygdala_corr,
        }

    return amygdala_correlations


# =============================================================================
# Within-subject change (baseline vs follow-up)
# =============================================================================
def analyze_session_changes(
        amygdala_correlations: dict,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
) -> pd.DataFrame:
    """
    For each (seed, region), compute paired t-test between follow-up and baseline across
    subjects that have both sessions.

    Returns:
        pd.DataFrame with columns:
        ['region','seed','mri1_mean','mri3_mean','mean_difference','t_statistic','p_value','n_subjects']
        (Column names keep historical naming; they represent baseline/follow-up means.)
    """
    # Resolve session labels (e.g., 'ses-MRI1', 'ses-MRI3')
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )
    print(f"Comparing {baseline_label} (baseline) vs {followup_label} (follow-up)")

    # Organize by session
    session_to_subjects = {baseline_label: {}, followup_label: {}}
    for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
        if session_label in session_to_subjects:
            session_to_subjects[session_label][subject_id] = seed_series_dict

    common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
        session_to_subjects[followup_label].keys()
    )
    print(f"Subjects with both sessions: {len(common_subjects)}")
    if len(common_subjects) < 1:
        print("Need at least 1 subject for comparison")
        return pd.DataFrame([])

    # Gather all regions available (union across all subjects in baseline)
    all_regions = set()
    for subject_id in common_subjects:
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_L"].index)
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_R"].index)

    print(f"Total unique regions found: {len(all_regions)}")
    print(f"Sample regions: {list(all_regions)[:5]}")

    results_rows = []
    for region_name in all_regions:
        for seed_name in ["Amygdala_L", "Amygdala_R"]:
            baseline_values = []
            followup_values = []

            for subject_id in common_subjects:
                baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
                followup_series = session_to_subjects[followup_label][subject_id][seed_name]
                if region_name in baseline_series and region_name in followup_series:
                    baseline_values.append(float(baseline_series[region_name]))
                    followup_values.append(float(followup_series[region_name]))

            if len(baseline_values) >= 1 and len(followup_values) >= 1:
                t_statistic, p_value = ttest_rel(baseline_values, followup_values)
                mean_difference = np.mean(followup_values) - np.mean(baseline_values)
                results_rows.append({
                    "region": region_name,
                    "seed": seed_name,
                    # Keep historical column names for continuity
                    "mri1_mean": np.mean(baseline_values),
                    "mri3_mean": np.mean(followup_values),
                    "mean_difference": mean_difference,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "n_subjects": len(baseline_values),
                })

    results_data_frame = pd.DataFrame(results_rows)
    return results_data_frame


def plot_session_changes(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Create bar plots for regions showing larger changes (or top-10 by |t| if none pass thresholds).
    """
    if results_data_frame is None or results_data_frame.empty:
        print("No results to plot")
        return

    significant_results = results_data_frame[
        (results_data_frame["p_value"] < 0.05) & (results_data_frame["t_statistic"].abs() > 2.0)
    ].copy()

    if significant_results.empty:
        print("No significant changes found (p < 0.05, |t| > 2.0), showing top 10 by |t|")
        significant_results = results_data_frame.reindex(
            results_data_frame["t_statistic"].abs().sort_values(ascending=False).index
        ).head(10).copy()

    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        seed_results = significant_results[significant_results["seed"] == seed_name].copy()
        if seed_results.empty:
            print(f"No data for {seed_name}")
            continue

        seed_results["abs_t_stat"] = seed_results["t_statistic"].abs()
        seed_results = seed_results.sort_values("abs_t_stat", ascending=False)

        # Two-panel figure: t-stats and -log10 p
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        colors = ["red" if val > 0 else "blue" for val in seed_results["t_statistic"]]
        ax1.barh(range(len(seed_results)), seed_results["t_statistic"], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(seed_results)))
        ax1.set_yticklabels(
            [r[:25] + "..." if len(r) > 25 else r for r in seed_results["region"]]
        )
        ax1.set_xlabel("T-Statistic")
        ax1.set_title(f"{seed_name} - T-Statistics (Follow-up vs Baseline)")
        ax1.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax1.grid(True, alpha=0.3)

        for i, (_, row) in enumerate(seed_results.iterrows()):
            p_text = f"p={row['p_value']:.3f}"
            ax1.text(row["t_statistic"], i, f" {p_text}", va="center", fontsize=8)

        ax2.barh(range(len(seed_results)), -np.log10(seed_results["p_value"]), color=colors, alpha=0.7)
        ax2.set_yticks(range(len(seed_results)))
        ax2.set_yticklabels(
            [r[:25] + "..." if len(r) > 25 else r for r in seed_results["region"]]
        )
        ax2.set_xlabel("-log10(p-value)")
        ax2.set_title(f"{seed_name} - Statistical Significance")
        ax2.axvline(x=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p=0.05")
        ax2.axvline(x=-np.log10(0.01), color="orange", linestyle="--", alpha=0.7, label="p=0.01")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figure_path = os.path.join(output_folder, f"{seed_name}_session_changes.png")
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        # Print quick summary
        print(f"\n=== {seed_name} Session Changes ===")
        print("Top 10 regions by absolute t-statistic:")
        for _, row in seed_results.head(10).iterrows():
            direction_text = "increased" if row["mean_difference"] > 0 else "decreased"
            print(f"  {row['region']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f}, {direction_text}")

    # Summary stats
    print("\n=== Summary Statistics ===")
    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        seed_rows = results_data_frame[results_data_frame["seed"] == seed_name]
        print(f"\n{seed_name}:")
        print(f"  Total regions tested: {len(seed_rows)}")
        print(f"  Significant changes (p < 0.05): {len(seed_rows[seed_rows['p_value'] < 0.05])}")
        print(f"  Large changes (|t| > 2.0): {len(seed_rows[seed_rows['t_statistic'].abs() > 2.0])}")
        print(f"  Mean t-statistic: {seed_rows['t_statistic'].mean():.3f}")
        print(f"  Mean p-value: {seed_rows['p_value'].mean():.3f}")

    results_csv = os.path.join(output_folder, "session_changes_results.csv")
    results_data_frame.to_csv(results_csv, index=False)
    print(f"\nSaved detailed results to: {results_csv}")


# =============================================================================
# Between-group deltas (follow-up - baseline) and volcano plot
# =============================================================================
def cohen_d_independent(group_x, group_y) -> float:
    """
    Cohen's d for independent samples (pooled SD).
    """
    array_x = np.asarray(group_x)
    array_y = np.asarray(group_y)
    n_x, n_y = len(array_x), len(array_y)
    if n_x < 2 or n_y < 2:
        return np.nan
    var_x, var_y = np.var(array_x, ddof=1), np.var(array_y, ddof=1)
    pooled_sd = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)) if (n_x + n_y - 2) > 0 else np.nan
    if pooled_sd is None or np.isnan(pooled_sd) or pooled_sd == 0:
        return np.nan
    return (np.mean(array_x) - np.mean(array_y)) / pooled_sd


def compute_subject_deltas(
        amygdala_correlations: dict,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
) -> dict:
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )

    # Index by session
    session_to_subjects = {baseline_label: {}, followup_label: {}}
    for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
        if session_label in session_to_subjects:
            session_to_subjects[session_label][subject_id] = seed_series_dict

    common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
        session_to_subjects[followup_label].keys()
    )
    if not common_subjects:
        raise ValueError("No subjects with both baseline and follow-up sessions.")

    # Collect all regions from baseline (union across subjects)
    all_regions = set()
    for subject_id in common_subjects:
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_L"].index)

    deltas_by_seed_region = {}
    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        for region_name in all_regions:
            key = (seed_name, region_name)
            deltas_by_seed_region[key] = {}
            for subject_id in common_subjects:
                baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
                followup_series = session_to_subjects[followup_label][subject_id][seed_name]
                if region_name in baseline_series and region_name in followup_series:
                    delta_value = float(followup_series[region_name]) - float(baseline_series[region_name])
                    deltas_by_seed_region[key][subject_id] = delta_value

    return deltas_by_seed_region


def between_group_tests(
        amygdala_correlations: dict,
        randomization_xlsx_path: str,
        output_folder: str,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
        subject_col: str = "SubID",
        group_col: str = "Group_Simbol",
) -> tuple:
    """
    Welch's t-test comparing (follow-up - baseline) between ketamine and control groups
    for each (seed, region). Also computes Cohen's d.

    Returns:
        (pd.DataFrame, bool): (results sorted by p-value, had_groups flag)
    """
    subject_to_group, _ = load_group_table(
        randomization_xlsx_path,
        subject_col=subject_col,
        group_col=group_col,
        ketamine_groups=KETAMINE_GROUP_SYMBOLS,
        control_groups=CONTROL_GROUP_SYMBOLS,
    )

    # Check presence of BOTH groups (auto-fallback trigger)
    group_values = set(v for v in subject_to_group.values() if v is not None)
    has_ket = "ketamine" in group_values
    has_ctrl = "control" in group_values
    had_groups = has_ket and has_ctrl

    if not had_groups:
        print("No valid ketamine/control group symbols found in the randomization file.")
        return pd.DataFrame([]), False

    deltas_by_seed_region = compute_subject_deltas(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )

    rows = []
    for (seed_name, region_name), subject_to_delta in deltas_by_seed_region.items():
        ketamine_deltas = []
        control_deltas = []
        for subject_id, delta_value in subject_to_delta.items():
            group_label = subject_to_group.get(subject_id)
            if group_label == "ketamine":
                ketamine_deltas.append(delta_value)
            elif group_label == "control":
                control_deltas.append(delta_value)

        if len(ketamine_deltas) >= 2 and len(control_deltas) >= 2:
            t_statistic, p_value = ttest_ind(ketamine_deltas, control_deltas, equal_var=False)
            cohens_d = cohen_d_independent(ketamine_deltas, control_deltas)
            rows.append({
                "seed": seed_name,
                "region": region_name,
                "n_ketamine": len(ketamine_deltas),
                "n_control": len(control_deltas),
                "mean_delta_ketamine": np.mean(ketamine_deltas),
                "mean_delta_control": np.mean(control_deltas),
                "mean_diff_(ket-control)": np.mean(ketamine_deltas) - np.mean(control_deltas),
                "t_statistic": t_statistic,
                "p_value": p_value,
                "cohens_d": cohens_d,
            })

    results_data_frame = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame([])
    if not results_data_frame.empty:
        out_path = os.path.join(output_folder, "between_group_followup_minus_baseline_amygdala.csv")
        results_data_frame.to_csv(out_path, index=False)
        print(f"Saved between-group results to: {out_path}")
    else:
        print("No regions had sufficient subjects in both groups for testing.")
    return results_data_frame, True


def volcano_plot(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Simple volcano plot: x = mean difference (ketamine - control), y = -log10(p).
    One PNG per seed.
    """
    if results_data_frame is None or results_data_frame.empty:
        return

    for seed_name in results_data_frame["seed"].unique():
        per_seed = results_data_frame[results_data_frame["seed"] == seed_name].copy()
        if per_seed.empty:
            continue
        per_seed["neglog10p"] = -np.log10(per_seed["p_value"])
        plt.figure(figsize=(8, 6))
        plt.scatter(per_seed["mean_diff_(ket-control)"], per_seed["neglog10p"])
        plt.axhline(-np.log10(0.05), linestyle="--")
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("Mean Δ difference (ketamine − control)")
        plt.ylabel("−log10 p")
        plt.title(f"{seed_name}: follow-up − baseline group difference")
        plt.tight_layout()
        fig_path = os.path.join(output_folder, f"volcano_{seed_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()


# =============================================================================
# Visualization on atlas (optional best-effort label matching)
# =============================================================================
def create_brain_visualization(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Attempt to visualize significant t-statistics per region on Harvard-Oxford atlas.
    Falls back to a bar plot if atlas mapping fails.
    """
    if results_data_frame is None or results_data_frame.empty:
        print("No results to visualize")
        return

    try:
        print("Loading Harvard-Oxford atlas...")
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_labels = atlas.labels  # list of strings

        # For each seed, construct a vector of t-stats over atlas labels
        for seed_name in ["Amygdala_L", "Amygdala_R"]:
            seed_rows = results_data_frame[results_data_frame["seed"] == seed_name].copy()
            if seed_rows.empty:
                print(f"No data for {seed_name}")
                continue

            # Prefer significant rows, else top-10 by |t|
            filtered = seed_rows[
                (seed_rows["p_value"] < 0.05) & (seed_rows["t_statistic"].abs() > 2.0)
            ].copy()
            if filtered.empty:
                print(f"No significant changes for {seed_name}, showing top 10 by |t|")
                filtered = seed_rows.reindex(
                    seed_rows["t_statistic"].abs().sort_values(ascending=False).index
                ).head(10).copy()

            # Fallback: ranked bar plot of regions by t-statistic
            fig, ax = plt.subplots(figsize=(12, 8))
            filtered_sorted = filtered.sort_values("t_statistic", ascending=True)
            colors = ["red" if x > 0 else "blue" for x in filtered_sorted["t_statistic"]]
            ax.barh(range(len(filtered_sorted)), filtered_sorted["t_statistic"], color=colors, alpha=0.7)
            ax.set_yticks(range(len(filtered_sorted)))
            ax.set_yticklabels(
                [r[:30] + "..." if len(r) > 30 else r for r in filtered_sorted["region"]]
            )
            ax.set_xlabel("T-Statistic")
            ax.set_title(f"{seed_name} - Brain Regions with Connectivity Changes")
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path_bar = os.path.join(
                output_folder, f"{seed_name.replace(' ', '_')}_connectivity_changes.png"
            )
            plt.savefig(path_bar, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"Created connectivity changes plot for {seed_name}")
    except Exception as error:
        print(f"Error creating brain visualization: {error}")
        print("This might be due to missing nilearn or atlas data")


# =============================================================================
# Main
# =============================================================================
def plot_between_group_top_regions(between_group_results):
    for seed_val, group_df in between_group_results.groupby("seed"):
        top_10_regions = group_df.sort_values("p_value").head(10).copy()
        colors = ["red" if diff > 0 else "blue" for diff in top_10_regions["mean_diff_(ket-control)"]]

        plt.figure(figsize=(10, 6))
        plt.barh(top_10_regions["region"], top_10_regions["p_value"], color=colors)
        plt.xlabel("p-value")
        plt.ylabel("Region")
        plt.xlim(0, 0.2)  # p-values always between 0 and 1
        plt.title(f"Top 10 Regions by Significance ({seed_val})")
        plt.gca().invert_yaxis()  # keeps the smallest p-values at the top
        plt.grid(True, axis="x", alpha=0.3)

        # Annotate bars with actual p-values
        for i, p in enumerate(top_10_regions["p_value"]):
            plt.text(-np.log10(p) + 0.05, i, f"p={p:.3e}", va="center", fontsize=8)

        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ketamine Δ > Control Δ'),
            Patch(facecolor='blue', label='Ketamine Δ < Control Δ')
        ]
        plt.legend(handles=legend_elements, title="Group difference", loc="lower right")

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_FOLDER, f"top10_between_group_{seed_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved plot for {seed_val} to: {output_path}")


if __name__ == "__main__":
    print("=== Amygdala Seed Connectivity Analysis ===")
    correlation_matrices = compute_pearson_correlations(PROJECT_ROOT)
    filtered_correlation_matrices = filter_correlation_matrices_by_fd_motion_threshold(
        report_path, correlation_matrices
    )
    if not correlation_matrices:
        print("No correlation matrices created. Check your data files.")
        raise SystemExit(1)

    amygdala_correlations = extract_amygdala_correlations(correlation_matrices)

    if ANALYZE_BY_GROUP:
        between_group_results, had_groups = between_group_tests(
            amygdala_correlations=amygdala_correlations,
            randomization_xlsx_path=RANDOMIZATION_XLSX_PATH,
            output_folder=OUTPUT_FOLDER,
            baseline_session_keywords=BASELINE_SESSION_KEYWORDS,
            followup_session_keywords=FOLLOWUP_SESSION_KEYWORDS,
            subject_col=RANDOMIZATION_SUBJECT_COLUMN,
            group_col=RANDOMIZATION_GROUP_COLUMN,
        )

        if had_groups and not between_group_results.empty:
            plot_between_group_top_regions(between_group_results)
        else:
            print("\nNo valid groups found or no results. Falling back to within-subject analysis...")
            within_df = analyze_session_changes(
                amygdala_correlations=amygdala_correlations,
                baseline_session_keywords=BASELINE_SESSION_KEYWORDS,
                followup_session_keywords=FOLLOWUP_SESSION_KEYWORDS,
            )
            plot_session_changes(within_df, OUTPUT_FOLDER)
    else:
        print("\nRunning within-subject follow-up vs baseline analysis (group analysis disabled)...")
        within_df = analyze_session_changes(
            amygdala_correlations=amygdala_correlations,
            baseline_session_keywords=BASELINE_SESSION_KEYWORDS,
            followup_session_keywords=FOLLOWUP_SESSION_KEYWORDS,
        )
        plot_session_changes(within_df, OUTPUT_FOLDER)

    print("\n=== Analysis Complete ===")
