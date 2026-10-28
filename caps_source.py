"""
caps_source.py — the ONE source for treatment arms and CAPS-5, both sites, in every analysis.

organized_results/behavioural_yale_tlv.csv  (site, subject, arm, timepoint, session, caps5_total)
  Built from yale/randomization.xlsx (Yale rows + its TAU rows = TLV, day-0 baseline), verified
  against tlv/RandomizationTable.xlsx (30/30 shared arms identical) and clinical/tlv_caps.csv.

Rules:
  • No active script reads randomization.xlsx, RandomizationTable.xlsx or tlv_caps.csv directly.
    `load_treatments()` in the analysis modules delegates here; check_single_source.py enforces it.
  • load_arms() refuses to run if any scanned subject of the site has no arm (the 2026-10-19 bug:
    5 TLV subjects were silently dropped from every TLV-only group test).
"""
import re
import pandas as pd
import kpe_config as cfg

CAPS_FILE = cfg.BASE_DIR / "organized_results" / "behavioural_yale_tlv.csv"
_SITE = {"tau": "tlv", "tlv": "tlv", "yale": "yale"}
ARMS = {"ket0.5", "ket0.2", "placebo"}


def subject_id(value) -> str:
    """'sub-010' / 'KPE1780' -> 'sub-010' / 'sub-1780' (same rule as the imaging file names)."""
    m = re.search(r"(\d+)", str(value))
    return f"sub-{int(m.group(1)):03d}" if m else str(value)


def load_arms(site: str, check_scans: bool = True) -> dict:
    """subject -> arm ('ket0.5' | 'ket0.2' | 'placebo') for 'tau' | 'yale' | 'combined'.

    Raises if a subject has two arms, appears under both sites, or (check_scans) has
    timeseries for this site but no arm."""
    d = pd.read_csv(CAPS_FILE)
    out = {}
    for s in (["tau", "yale"] if site == "combined" else [site]):
        per = d[(d["site"] == _SITE[s]) & d["arm"].isin(ARMS)].groupby("subject")["arm"].unique()
        two = {k: list(v) for k, v in per.items() if len(v) > 1}
        if two:
            raise ValueError(f"[{s}] subjects with two arms in {CAPS_FILE.name}: {two}")
        arms = {subject_id(k): v[0] for k, v in per.items()}
        clash = set(arms) & set(out)
        if clash:
            raise ValueError(f"subject ids under both sites: {sorted(clash)}")
        if check_scans:
            scanned = {subject_id(re.search(r"sub-[A-Za-z0-9]+", p.name).group(0))
                       for p in cfg.find_ts_files(cfg.DATASETS[s]["anatomical"], "tian_s2")}
            missing = sorted(scanned - set(arms))
            if missing:
                raise ValueError(f"[{s}] scanned subjects with NO arm in {CAPS_FILE.name}: {missing}. "
                                 "Fix the table; never drop them silently.")
        out.update(arms)
    return out


def arms_for(rand_path_or_list) -> dict:
    """Drop-in for the old load_treatments(cfg.DATASETS[ds]['randomization']): works out which
    dataset was meant and returns load_arms() for it."""
    for ds in ("tau", "yale", "combined"):
        if cfg.DATASETS[ds]["randomization"] == rand_path_or_list:
            return load_arms(ds)
    raise ValueError(f"unknown randomization argument: {rand_path_or_list!r}")


def load_dcaps(site: str, base: str = "baseline", follow: str = "30day") -> dict:
    """ΔCAPS = CAPS(follow) − CAPS(base) per subject; negative = improvement.

    site: 'tau' | 'yale' | 'combined'. combined = union of both sites
    (raises if a subject id appears under both, so one site can never overwrite the other).
    """
    if site == "combined":
        tau, yale = load_dcaps("tau", base, follow), load_dcaps("yale", base, follow)
        overlap = set(tau) & set(yale)
        if overlap:
            raise ValueError(f"CAPS subject ids under both sites: {sorted(overlap)}")
        return {**tau, **yale}
    d = pd.read_csv(CAPS_FILE)
    d = d[d["site"] == _SITE[site]]
    piv = d.pivot_table(index="subject", columns="timepoint", values="caps5_total", aggfunc="first")
    if base not in piv.columns or follow not in piv.columns:
        return {}
    dc = (piv[follow] - piv[base]).dropna()
    return {subject_id(s): float(v) for s, v in dc.items()}
