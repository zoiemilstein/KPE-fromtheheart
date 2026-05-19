from pathlib import Path
import re


BASE_DIR = Path("/Users/zoiemilstein/רפואה/מעבדה/kpe")
if not BASE_DIR.exists():
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

TIMESERIES_DIR = BASE_DIR / "timeseries"

ANATOMICAL_TS_DIR = TIMESERIES_DIR / "anatomical"
GLOBAL_TS_DIR = TIMESERIES_DIR / "global"

ANATOMICAL_SCRUB_CSV = ANATOMICAL_TS_DIR / "scrubbing_report_filtered.csv"
GLOBAL_SCRUB_CSV = GLOBAL_TS_DIR / "scrubbing_report_filtered.csv"

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