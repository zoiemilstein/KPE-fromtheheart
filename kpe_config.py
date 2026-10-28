from pathlib import Path
import re


import os as _os
BASE_DIR = Path(_os.environ.get("KPE_BASE_DIR", "/Users/zoiemilstein/רפואה/מעבדה/kpe"))

# Canonical time-series data (reorganized from Amir's yale+tlv dump, 2026-08-25).
# Clean per-site layout: {site}/{anatomical|global}/ holding all atlases
# (schaefer400, tian_s2, harvard_oxford_cortical) + one scrubbing_report_filtered.csv
# + the site's randomization table.
TAU_DIR = BASE_DIR / "timeseries_2026-08" / "tlv"      # TLV ("tau") site
YALE_DIR = BASE_DIR / "timeseries_2026-08" / "yale"    # Yale site


if not BASE_DIR.exists():
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

if not TAU_DIR.exists():
    raise FileNotFoundError(f"TAU_DIR does not exist: {TAU_DIR}")

if not YALE_DIR.exists():
    raise FileNotFoundError(f"YALE_DIR does not exist: {YALE_DIR}")


OUTPUT_DATA_DIR = BASE_DIR / "output_data"

# March 2026 TLV extraction (6 mm smoothing), anatomical (no GSR). Used only by dfc_poster/.
# Override with the KPE_TIMESERIES_6MM environment variable.
TIMESERIES_6MM = Path(_os.environ.get("KPE_TIMESERIES_6MM",
                                      str(Path.home() / "Downloads" / "timeseries" / "anatomical")))
RESULTS_DIR = BASE_DIR / "results"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_entities(filename: str):
    parsed_dict = {}

    patterns = {
        "subject": r"(sub-[A-Za-z0-9]+)",
        "session": r"(ses-[A-Za-z0-9]+)",
        "task": r"(task-[A-Za-z0-9]+)",
        "run": r"(run-[A-Za-z0-9]+)",
        "acq": r"(acq-[A-Za-z0-9]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, filename)
        if m:
            parsed_dict[key] = m.group(1)

    return parsed_dict


def find_ts_files(root, atlas_tag: str):
    """Find timeseries CSVs for an atlas under one folder OR several.

    `root` may be a single path (tau / yale) or a list of paths — the
    'combined' dataset is tau + yale read together. Returns a sorted list
    of all matching files across the given root(s).
    """
    roots = root if isinstance(root, (list, tuple)) else [root]
    files = []
    for r in roots:
        files.extend(Path(r).rglob(f"*{atlas_tag}_ts.csv"))
    return sorted(files)


def find_randomization_file(root: Path):
    """
    Finds the first Excel randomization table under a dataset folder.
    The file name must contain the word 'randomization'.
    """

    files = sorted(
        f for f in root.rglob("*.xlsx")
        if "randomization" in f.name.lower() and not f.name.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(
            f"No randomization Excel file found under: {root}\n"
            f"Expected an .xlsx file with 'randomization' in the filename."
        )

    if len(files) > 1:
        print("\nWARNING: More than one randomization file found:")
        for f in files:
            print(f"  {f}")
        print(f"\nUsing first one:\n  {files[0]}")

    return files[0]


TAU_RAND = find_randomization_file(TAU_DIR)
YALE_RAND = find_randomization_file(YALE_DIR)

DATASETS = {
    "tau": {
        "root": TAU_DIR,
        "anatomical": TAU_DIR / "anatomical",
        "global": TAU_DIR / "global",
        "anatomical_scrub": TAU_DIR / "anatomical" / "scrubbing_report_filtered.csv",
        "global_scrub": TAU_DIR / "global" / "scrubbing_report_filtered.csv",
        "randomization": TAU_RAND,
    },

    "yale": {
        "root": YALE_DIR,
        "anatomical": YALE_DIR / "anatomical",
        "global": YALE_DIR / "global",
        "anatomical_scrub": YALE_DIR / "anatomical" / "scrubbing_report_filtered.csv",
        "global_scrub": YALE_DIR / "global" / "scrubbing_report_filtered.csv",
        "randomization": YALE_RAND,
    },

    # 'combined' = tau + yale read together. No separate folder, no merged files:
    # the list-valued paths are consumed by find_ts_files() and load_group_table(),
    # which both accept a list of sources.
    "combined": {
        "root": [TAU_DIR, YALE_DIR],
        "anatomical": [TAU_DIR / "anatomical", YALE_DIR / "anatomical"],
        "global": [TAU_DIR / "global", YALE_DIR / "global"],
        "anatomical_scrub": [TAU_DIR / "anatomical" / "scrubbing_report_filtered.csv",
                             YALE_DIR / "anatomical" / "scrubbing_report_filtered.csv"],
        "global_scrub": [TAU_DIR / "global" / "scrubbing_report_filtered.csv",
                         YALE_DIR / "global" / "scrubbing_report_filtered.csv"],
        "randomization": [TAU_RAND, YALE_RAND],
    },
}


# Optional alias specifically for Schaefer scripts
SCHAEFER_DATASETS = DATASETS

# --------------------------------------------------------------------------- #
# MOTION / LENGTH RULES  (ONE place; every analysis gets its scan filter from scan_drop())
#   1. drop a scan if scrubbed_volumes > SCRUB_THRESHOLD
#   2. drop a scan if retained volumes (total_vols - scrubbed_volumes) < MIN_RETAINED_VOLUMES
#      (TR = 1 s, so 300 volumes = 5 min of clean data; docs/research/2026-10-20_methods_decisions.md)
# Duplicate rest runs: rule 2 is judged on the run that is analysed (rest2 wins); rule 1 keeps the
# older "any row of the session" behaviour.
# --------------------------------------------------------------------------- #
SCRUB_THRESHOLD = 115
MIN_RETAINED_VOLUMES = 300
MOTION_RULE_TEXT = (f"Scan exclusion: scrubbed_volumes > {SCRUB_THRESHOLD} dropped; AND retained volumes "
                    f"(total_vols - scrubbed_volumes) < {MIN_RETAINED_VOLUMES} dropped (TR=1 s, >= 5 min of clean data; "
                    f"length judged on the analysed run, rest2 wins). Rule lives in kpe_config.scan_drop().")


def scan_drop(scrub_path_or_list, normalize=None, with_reasons=False):
    """Set of (subject, session) to DROP, from scrubbing_report_filtered.csv path(s)."""
    import pandas as _pd
    paths = scrub_path_or_list if isinstance(scrub_path_or_list, (list, tuple)) else [scrub_path_or_list]
    norm = normalize or (lambda v: str(v))
    drop, reasons = set(), {}
    for p in paths:
        if not p or not _os.path.exists(p):
            continue
        d = _pd.read_csv(p)
        if "scrubbed_volumes" not in d.columns:
            continue
        d["_sub"] = d["subject"].map(norm); d["_ses"] = d["session"].astype(str)
        for _, r in d[d["scrubbed_volumes"] > SCRUB_THRESHOLD].iterrows():
            drop.add((r["_sub"], r["_ses"])); reasons[(r["_sub"], r["_ses"])] = "motion"
        if "total_vols" in d.columns:
            d["_ret"] = d["total_vols"] - d["scrubbed_volumes"]
            for key, g in d.sort_values("task").groupby(["_sub", "_ses"]):
                last = g.iloc[-1]  # rest2 sorts after rest1
                if last["_ret"] < MIN_RETAINED_VOLUMES:
                    drop.add(key); reasons.setdefault(key, "short")
    return (drop, reasons) if with_reasons else drop
