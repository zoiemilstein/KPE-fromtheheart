import os
import re
import sys
import datetime
from typing import Dict, List, Tuple
import nibabel as nib
import pandas as pd
from nilearn import datasets
import nilearn

print(f"Nilearn version: {nilearn.__version__}")
from nilearn.maskers import NiftiLabelsMasker

# -------------------------------------------------------------------------
# --- parameters ----------------------------------------------------------
# -------------------------------------------------------------------------
STANDARTIZE: str = "zscore_sample"
SMOOTHING_FWHM: float = 8
DETREND: bool = True
HIGH_PASS: float = 0.01
LOW_PASS: float = 0.08
T_R: float = 1.0
NUM_VOLS_TO_REMOVE: int = 4
DO_SCRUBBING: bool = True
conf_cols = [
    "trans_x", "trans_y", "trans_z",
    "global_signal",
    "rot_x", "rot_y", "rot_z",
    "std_dvars", "framewise_displacement",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05",
]
PROJECT_ROOT = r"D:\amir_shared_folder\fmriprep_072025"
OUTPUT_SUFFIX = "_aal_ts.csv"
SCRUB_REPORT_CSV = "scrubbing_report.csv"

os.environ.setdefault("NILEARN_DATA", r"C:\Users\amirh\Documents\nilearn_cache")

# Thresholds for motion warnings
FD_THRESHOLD = 0.5
DVARS_THRESHOLD = 1.5
scrubbed_volumes_threshold = 0.2

# For reporting
scrub_stats = []


# -------------------------------------------------------------------------
# --- helpers -------------------------------------------------------------
# -------------------------------------------------------------------------

def remove_first_n_volumes(nifti_path: str, n_vol: int) -> nib.Nifti1Image:
    img = nib.load(nifti_path)
    data = img.get_fdata()[..., n_vol:]
    return nib.Nifti1Image(data, affine=img.affine, header=img.header)


def get_aal_atlas() -> Tuple[List[str], str]:
    aal = datasets.fetch_atlas_aal()
    return aal.labels, aal.maps


class FMRIFileSet:
    def __init__(self, subject: str, session: str, bold_path: str, confounds_path: str):
        self.subject = subject
        self.session = session
        self.bold_path = bold_path
        self.confounds_path = confounds_path

    def __repr__(self):
        return (f"Subject: {self.subject}, Session: {self.session}\n"
                f"BOLD: {self.bold_path}\n"
                f"Confounds: {self.confounds_path}")



def discover_files(project_root: str) -> List[FMRIFileSet]:
    file_sets: List[FMRIFileSet] = []
    for subj in sorted(p for p in os.listdir(project_root) if p.startswith("sub-")):
        subj_path = os.path.join(project_root, subj)
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue
            bold = conf = None
            for fname in os.listdir(func_dir):
                if "rest" in fname and "preproc_bold" in fname and "MNI" in fname and fname.endswith(".nii.gz"):
                    bold = os.path.join(func_dir, fname)
                elif "rest" in fname and fname.endswith("confounds_timeseries.tsv"):
                    conf = os.path.join(func_dir, fname)
            if bold and conf:
                file_sets.append(FMRIFileSet(subj, ses, bold, conf))
    return file_sets


def extract_time_series(img: nib.Nifti1Image, labels_img, labels: List[str], conf_df: pd.DataFrame) -> pd.DataFrame:
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=STANDARTIZE,
        smoothing_fwhm=SMOOTHING_FWHM,
        detrend=DETREND,
        standardize_confounds=True,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        t_r=T_R,
        memory="nilearn_cache",
        verbose=0,
    )
    time_series = masker.fit_transform(img, confounds=conf_df)
    return pd.DataFrame(time_series, columns=labels)


def build_confounds(conf_path: str, subject: str, session: str) -> pd.DataFrame:
    df = pd.read_csv(conf_path, sep="\t")
    df = df.loc[NUM_VOLS_TO_REMOVE:].reset_index(drop=True)
    motion_df = df[["framewise_displacement", "std_dvars"]].copy()

    fd_mean = motion_df["framewise_displacement"].mean()
    dvars_mean = motion_df["std_dvars"].mean()
    fd_max = motion_df["framewise_displacement"].max()
    dvars_max = motion_df["std_dvars"].max()

    scrubbed_volumes = 0
    conf_df = df[conf_cols].copy()

    if DO_SCRUBBING:
        spike_mask = motion_df["framewise_displacement"] > FD_THRESHOLD
        scrubbed_volumes = spike_mask.sum()
        if scrubbed_volumes > 0:
            spike_regressors = pd.DataFrame(0, index=motion_df.index,
                                            columns=[f"motion_spike_{i}" for i in range(scrubbed_volumes)])
            for idx, timepoint in enumerate(spike_mask[spike_mask].index):
                spike_regressors.iloc[timepoint, idx] = 1
            conf_df = pd.concat([conf_df, spike_regressors], axis=1)
            print(f"{subject} {session} | Scrubbing applied to {scrubbed_volumes} volumes")
        else:
            print(f"{subject} {session} | No scrubbing needed")
    else:
        print(f"{subject} {session} | Scrubbing disabled")

    print(
        f"{subject} {session} | FD mean: {fd_mean:.3f}, max: {fd_max:.3f} | DVARS mean: {dvars_mean:.3f}, max: {dvars_max:.3f}")
    if fd_mean > FD_THRESHOLD:
        print(f"High motion detected (FD > {FD_THRESHOLD}) in {subject} {session}")
    if dvars_mean > DVARS_THRESHOLD:
        print(f"High DVARS detected (DVARS > {DVARS_THRESHOLD}) in {subject} {session}")

    scrub_stats.append({
        "subject": subject,
        "session": session,
        "fd_mean": fd_mean,
        "fd_max": fd_max,
        "dvars_mean": dvars_mean,
        "dvars_max": dvars_max,
        "scrubbed_volumes": scrubbed_volumes,
    })

    return conf_df


def create_time_series(project_root: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    labels, atlas_img = get_aal_atlas()
    time_series_dict: Dict[Tuple[str, str], pd.DataFrame] = {}

    for fs in discover_files(project_root):
        print(f"Processing {fs}")
        img_sliced = remove_first_n_volumes(fs.bold_path, NUM_VOLS_TO_REMOVE)
        conf_df = build_confounds(fs.confounds_path, fs.subject, fs.session)
        # Check scrubbed_volumes threshold
        if scrub_stats[-1]["scrubbed_volumes"] > (scrubbed_volumes_threshold * 580):
            print(f"Skipping {fs.subject} {fs.session} due to excessive scrubbing ({scrub_stats[-1]['scrubbed_volumes']} volumes)")
            continue
        df = extract_time_series(img_sliced, atlas_img, labels, conf_df)

        out_name = f"{fs.subject}_{fs.session}{OUTPUT_SUFFIX}"
        df.to_csv(out_name, index=False)
        time_series_dict[(fs.subject, fs.session)] = df
        print(f"Saved {out_name} | shape={df.shape}")

    return time_series_dict


def load_cached_time_series(csv_dir: str = ".") -> Dict[Tuple[str, str], pd.DataFrame]:
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for fname in os.listdir(csv_dir):
        if fname.endswith(OUTPUT_SUFFIX):
            subj, ses, *_ = fname.split("_")
            df = pd.read_csv(os.path.join(csv_dir, fname))
            cache[(subj, ses)] = df
            print(f"Loaded {fname} | shape={df.shape}")
    return cache


# -------------------------------------------------------------------------
# --- main ----------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == "__main__":
    regenerate = len(sys.argv) == 1 or sys.argv[1].lower() != "cache"
    if regenerate:
        print("Regenerating time series from scratch …")
        ts_dict = create_time_series(PROJECT_ROOT)
    else:
        print("Loading cached CSV files …")
        ts_dict = load_cached_time_series()

    print(f"Done. {len(ts_dict)} subject/session pairs processed.")

    if scrub_stats:
        print("\nScrubbing Summary Report:")
        summary_df = pd.DataFrame(scrub_stats)
        print(summary_df.to_string(index=False))

        summary_df["total_volumes"] = 580
        summary_df["scrub_percent"] = (summary_df["scrubbed_volumes"] / 580 * 100).round(1)
        summary_df["fd_threshold"] = FD_THRESHOLD
        summary_df["dvars_threshold"] = DVARS_THRESHOLD
        summary_df["smoothing_fwhm"] = SMOOTHING_FWHM
        summary_df["date"] = datetime.datetime.now().strftime("%Y-%m-%d")

        summary_df["confound_columns"] = ", ".join(conf_cols)
        summary_df[
            "nilearn_masker_config"] = f"standardize={STANDARTIZE}, smoothing_fwhm={SMOOTHING_FWHM}, detrend={DETREND}, low_pass={LOW_PASS}, high_pass={HIGH_PASS}, t_r={T_R}, standardize_confounds=True"
        summary_df.to_csv(SCRUB_REPORT_CSV, index=False)
        print(f"Report saved to {SCRUB_REPORT_CSV}")

        # Save general metadata separately
        with open("scrubbing_metadata.txt", "w") as f:
            f.write("Scrubbing Metadata Summary\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"FD threshold: {FD_THRESHOLD}\n")
            f.write(f"DVARS threshold: {DVARS_THRESHOLD}\n")
            f.write(f"Total volumes per scan: 580\n")
            f.write(f"Confounds used:\n  {', '.join(conf_cols)}\n")
            f.write("Nilearn Masker config:\n")
            f.write(f"  standardize = {STANDARTIZE}\n")
            f.write(f"  smoothing_fwhm = {SMOOTHING_FWHM}\n")
            f.write(f"  detrend = {DETREND}\n")
            f.write(f"  low_pass = {LOW_PASS}\n")
            f.write(f"  high_pass = {HIGH_PASS}\n")
            f.write(f"  t_r = {T_R}\n")
            f.write("  standardize_confounds = True\n")
