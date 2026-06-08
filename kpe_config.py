from pathlib import Path
import re


BASE_DIR = Path("/Users/zoiemilstein/רפואה/מעבדה/kpe")

TAU_DIR = BASE_DIR / "timeseries"
YALE_DIR = BASE_DIR / "Yale-results-3sessons (1)"



if not BASE_DIR.exists():
    raise FileNotError(f"BASE_DIR does not exist: {BASE_DIR}")

if not TAU_DIR.exists():
    raise FileNotFoundError(f"TAU_DIR does not exist: {TAU_DIR}")

if not YALE_DIR.exists():
    raise FileNotFoundError(f"YALE_DIR does not exist: {YALE_DIR}")



OUTPUT_DATA_DIR = BASE_DIR / "output_data"
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


def find_ts_files(root: Path, atlas_tag: str):
    return sorted(
        root.rglob(f"*{atlas_tag}_ts.csv")
    )


def find_randomization_file(root: Path):
    """
    Finds the first Excel randomization table under a dataset folder.
    The file name must contain the word 'randomization'.
    """

    files = sorted(
        f for f in root.rglob("*.xlsx")
        if "randomization" in f.name.lower()
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


DATASETS = {
    "tau": {
        "root": TAU_DIR,
        "anatomical": TAU_DIR / "anatomical",
        "global": TAU_DIR / "global",
        "anatomical_scrub": TAU_DIR / "anatomical" / "scrubbing_report_filtered.csv",
        "global_scrub": TAU_DIR / "global" / "scrubbing_report_filtered.csv",
        "randomization": find_randomization_file(TAU_DIR),
    },

    "yale": {
        "root": YALE_DIR,
        "anatomical": YALE_DIR / "anatomical",
        "global": YALE_DIR / "global",
        "anatomical_scrub": YALE_DIR / "anatomical" / "scrubbing_report_filtered.csv",
        "global_scrub": YALE_DIR / "global" / "scrubbing_report_filtered.csv",
        "randomization": find_randomization_file(YALE_DIR),
    },
}


# Optional alias specifically for Schaefer scripts
SCHAEFER_DATASETS = DATASETS