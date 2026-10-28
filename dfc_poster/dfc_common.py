"""Shared helpers for the dFC-variance poster analyses (TLV site, ses-1, no-GSR, rest).

Pipeline: ROI/network-mean timeseries -> 60-TR sliding window (step 1) -> Pearson r per window
-> variance of the windowed r (ddof=1) = "dFC variance". Baseline dFC variance is related to
CAPS improvement (baseline - day 30; positive = better).

Paths come from kpe_config (4 mm = timeseries_2026-08/tlv/anatomical, 6 mm = TIMESERIES_6MM).
Arms and CAPS come only from caps_source. Run scripts from this folder or the repo root; the
repo root is added to sys.path below.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kpe_config as cfg          # noqa: E402
import caps_source as cs          # noqa: E402
from network_level_fc import schaefer_relabel, net_of   # noqa: E402

WINDOW = 60                 # TRs (TR = 1 s)
MIN_RETAINED = 120          # volumes kept after scrubbing, required for dFC
SCRUB_MAX = cfg.SCRUB_THRESHOLD   # 115: scans with scrubbed_volumes > 115 are dropped
EXCLUDE_REASON = {
    "sub-540": "excluded per protocol",
    "sub-576": "excluded per protocol",
    "sub-807": "excluded per protocol",
    "sub-949": "excluded per protocol",
    "sub-900": "randomized but never scanned: no TLV imaging at all (exclusion is a no-op, kept for the record)",
}
EXCLUDE = set(EXCLUDE_REASON)
ARMS = ["ket0.5", "ket0.2", "placebo"]          # 'placebo' = midazolam
ARM_LABEL = {"ket0.5": "ket0.5", "ket0.2": "ket0.2", "placebo": "midazolam"}
AMY_L = ["lAMY-lh", "mAMY-lh"]
AMY_R = ["lAMY-rh", "mAMY-rh"]
NET_CODE = {"Salience": "SalVentAttn", "DMN": "Default", "Control": "Cont"}
N_PERM = 200000
SEED = 2026


# ----------------------------------------------------------------------------- data loading
def data_dir(version: str) -> Path:
    """'4mm' -> 2026-08 TLV anatomical; '6mm' -> March extraction (kpe_config.TIMESERIES_6MM)."""
    if version == "4mm":
        return Path(cfg.TAU_DIR) / "anatomical"
    if version == "6mm":
        return Path(cfg.TIMESERIES_6MM)
    raise ValueError(version)


def scrub_table(version: str) -> pd.DataFrame:
    """ses-1 row per subject from scrubbing_report_filtered.csv (index = subject).
    4 mm has rest1/rest2 duplicates: the later file (rest2) wins. 6 mm session is named ses-MRI1."""
    t = pd.read_csv(data_dir(version) / "scrubbing_report_filtered.csv")
    if version == "6mm":
        return t[t.session == "ses-MRI1"].set_index("subject")
    return t[t.session == "ses-1"].sort_values("task").groupby("subject").tail(1).set_index("subject")


def ts_file(version: str, sub: str, atlas: str, scrub: pd.DataFrame):
    """Path of '<atlas>' timeseries ('schaefer400' | 'tian_s2') for the subject's ses-1 scan, or None."""
    d = data_dir(version)
    if version == "6mm":
        p = d / f"{sub}_ses-MRI1_task-rest_run-1_{atlas}_ts.csv"
        return p if p.exists() else None
    hits = glob.glob(str(d / f"{sub}_ses-1_task-{scrub.loc[sub, 'task']}_*{atlas}_ts.csv"))
    return Path(hits[0]) if hits else None


def scan_ok(scrub: pd.DataFrame, sub: str) -> bool:
    """Motion rule (scrubbed <= 115) plus >= 120 retained volumes."""
    if sub not in scrub.index:
        return False
    r = scrub.loc[sub]
    return bool(r.scrubbed_volumes <= SCRUB_MAX and (r.total_vols - r.scrubbed_volumes) >= MIN_RETAINED)


def improvement() -> dict:
    """subject -> CAPS improvement = baseline - day 30 (= -load_dcaps); positive = better."""
    return {k: -v for k, v in cs.load_dcaps("tau").items()}


def tlv_arms() -> dict:
    return cs.load_arms("tau")


def eligible_subjects(version: str):
    """Returns (scrub_table, kept_subjects, dropped_for_motion, no_scan). Kept have arm + CAPS, are not
    excluded and pass the motion / retained-volume rule. no_scan = subjects with arm + CAPS (not excluded)
    that have no ses-1 row in the scrubbing report."""
    scrub = scrub_table(version)
    arms, imp = tlv_arms(), improvement()
    cand = sorted(s for s in scrub.index if s in arms and s in imp and s not in EXCLUDE)
    kept = [s for s in cand if scan_ok(scrub, s)]
    no_scan = sorted(s for s in arms if s in imp and s not in EXCLUDE and s not in scrub.index)
    return scrub, kept, [s for s in cand if s not in kept], no_scan


# ----------------------------------------------------------------------------- ROI signals
def roi_signals(sch_path, tian_path) -> dict:
    """ROI / network-mean timeseries for one scan.
    Schaefer goes through schaefer_relabel() (the 6 mm header is already official and passes through).
    Keys: Salience, DMN, Control (Schaefer 7-net means); AmygL, AmygR (Tian S2 lAMY+mAMY per hemisphere)."""
    sch = schaefer_relabel(pd.read_csv(sch_path))
    tian = pd.read_csv(tian_path)
    out = {k: sch[[c for c in sch.columns if net_of(c) == code]].mean(axis=1).values
           for k, code in NET_CODE.items()}
    out["AmygL"] = tian[AMY_L].mean(axis=1).values
    out["AmygR"] = tian[AMY_R].mean(axis=1).values
    return out


# ----------------------------------------------------------------------------- dFC
def dfc_variance(x, y, window: int = WINDOW) -> float:
    """Variance (ddof=1) of the Pearson r in sliding windows of `window` TRs, step 1."""
    r = np.array([np.corrcoef(x[i:i + window], y[i:i + window])[0, 1]
                  for i in range(len(x) - window + 1)])
    return float(r.var(ddof=1))


def build_subject_table(version: str, pairs: dict):
    """Per-subject dFC variance. pairs = {'Salience-Control': ('Salience','Control'), ...}.
    Returns (DataFrame[version, subject, arm, pair, var, improvement, fd], info dict with counts)."""
    scrub, kept, dropped, no_scan = eligible_subjects(version)
    arms, imp = tlv_arms(), improvement()
    rows, missing = [], []
    for s in kept:
        sp, tp = ts_file(version, s, "schaefer400", scrub), ts_file(version, s, "tian_s2", scrub)
        if sp is None or tp is None:
            missing.append(s)
            continue
        sig = roi_signals(sp, tp)
        for name, (a, b) in pairs.items():
            rows.append(dict(version=version, subject=s, arm=arms[s], pair=name,
                             var=dfc_variance(sig[a], sig[b]), improvement=imp[s],
                             fd=scrub.loc[s, "fd_mean_filtered"]))
    df = pd.DataFrame(rows)
    n_arm = df.drop_duplicates("subject").arm.value_counts().to_dict() if len(df) else {}
    return df, dict(version=version, used=len(kept) - len(missing), dropped_motion=dropped,
                    missing_files=missing, per_arm=n_arm, arm_caps_no_scan=no_scan)


# ----------------------------------------------------------------------------- stats
def center_within_arm(values: pd.Series, arm: pd.Series) -> pd.Series:
    """Subtract each arm's mean."""
    return values - values.groupby(arm.values).transform("mean")


def corr_both(x, y) -> dict:
    """Spearman and Pearson (rho/r and p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sr, sp = stats.spearmanr(x, y)
    pr, pp = stats.pearsonr(x, y)
    return dict(n=len(x), spearman_rho=sr, spearman_p=sp, pearson_r=pr, pearson_p=pp)


def pooled_centered(d: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """All arms pooled: centre variance and improvement within arm, Spearman rho on the centred
    values; p = within-arm permutation (shuffle improvement within each arm, two-sided,
    (hits+1)/(n_perm+1)). Pearson on the centred values given as a secondary number."""
    d = d.reset_index(drop=True)
    v = center_within_arm(d["var"], d.arm).values
    y = center_within_arm(d.improvement, d.arm).values
    rho = stats.spearmanr(v, y)[0]
    rng = np.random.default_rng(seed)
    groups = [np.where(d.arm.values == a)[0] for a in d.arm.unique()]
    hits = 0
    for _ in range(n_perm):
        yp = d.improvement.values.copy()
        for ix in groups:
            yp[ix] = yp[rng.permutation(ix)]
        yc = center_within_arm(pd.Series(yp), d.arm).values
        if abs(stats.spearmanr(v, yc)[0]) >= abs(rho) - 1e-12:
            hits += 1
    pr, pp = stats.pearsonr(v, y)
    return dict(n=len(d), rho_centred=rho, p_perm=(hits + 1) / (n_perm + 1), pearson_r_centred=pr,
                pearson_p_centred=pp, n_perm=n_perm, seed=seed)


def per_arm_stats(d: pd.DataFrame) -> list:
    """Spearman + Pearson of variance vs improvement inside each arm (needs n >= 3)."""
    out = []
    for arm in ARMS:
        g = d[d.arm == arm]
        if len(g) >= 3:
            out.append(dict(arm=ARM_LABEL[arm], **corr_both(g["var"], g.improvement)))
    return out


def interaction(d: pd.DataFrame, arm_a: str, arm_b: str) -> dict:
    """Two-arm OLS: improvement ~ var * arm. Returns the var:arm interaction coefficient and p
    (arm_a coded 1, arm_b coded 0)."""
    g = d[d.arm.isin([arm_a, arm_b])].assign(grp=lambda x: (x.arm == arm_a).astype(int))
    m = smf.ols("improvement ~ grp * var", g).fit()
    return dict(contrast=f"{ARM_LABEL[arm_a]} vs {ARM_LABEL[arm_b]}", n=int(m.nobs),
                coef=m.params["grp:var"], p=m.pvalues["grp:var"])


# ----------------------------------------------------------------------------- run plumbing
def default_out(name: str) -> Path:
    return Path(cfg.OUTPUT_DATA_DIR) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"


def finish(out: Path, per_subject: pd.DataFrame, results: pd.DataFrame, methods: str):
    """Write per_subject.csv, results.csv and METHODS.txt."""
    out.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(out / "per_subject.csv", index=False)
    results.to_csv(out / "results.csv", index=False)
    (out / "METHODS.txt").write_text(methods)


def run_checks(checks: list) -> bool:
    """checks = [(label, value, expected, decimals)]. Compares round(value, decimals) to expected."""
    ok = True
    for label, val, exp, dec in checks:
        good = round(float(val), dec) == round(float(exp), dec)
        ok &= good
        print(f"  [{'OK  ' if good else 'FAIL'}] {label}: got {float(val):.{dec + 2}f}, poster {exp}")
    return ok


def common_methods(extra: str, n_perm=None) -> str:
    perm = (f"Pooled test: variance and improvement centred within arm; Spearman rho on centred values; p from {n_perm} within-arm permutations of improvement (seed {SEED}, two-sided, (hits+1)/(N+1)).\n"
            if n_perm else "")
    return f"""TLV site, ses-1 (baseline), anatomical (no GSR) timeseries, resting state.
dFC variance: ROI/network-mean timeseries, 60-TR sliding window (step 1), Pearson r per window, variance (ddof=1) of the windowed r. Scrubbed volumes are not censored.
Outcome: CAPS improvement = baseline - day 30 (positive = better), caps_source.load_dcaps('tau'); arms from caps_source.load_arms ('placebo' = midazolam).
Exclusions: {EXCLUDE_REASON}; scans with scrubbed_volumes > {SCRUB_MAX} dropped; fewer than {MIN_RETAINED} retained volumes dropped; duplicate rest runs: rest2 wins.
Data: 4 mm = {data_dir('4mm')}; 6 mm (March extraction) = {data_dir('6mm')}.
Schaefer-400 7-network means via schaefer_relabel(); Tian S2 amygdala = lAMY + mAMY per hemisphere.
{perm}Post hoc / exploratory analysis; no FDR recomputed here.
{extra}"""


# ----------------------------------------------------------------------------- static FC (result 4)
HO_VMPFC = "Frontal Medial Cortex"      # Harvard-Oxford, single midline label


def parcel_edge_fc(tian: pd.DataFrame, ho: pd.DataFrame, seeds=AMY_L, target=HO_VMPFC) -> float:
    """Parcel-edge FC: Pearson per (seed parcel, target) pair -> Fisher z -> mean over pairs."""
    assert len(tian) == len(ho), f"Tian ({len(tian)}) and HO ({len(ho)}) timeseries differ in length"
    zs = [np.arctanh(np.clip(np.corrcoef(tian[s].values, ho[target].values)[0, 1], -.9999, .9999))
          for s in seeds]
    return float(np.mean(zs))


def static_fc_table(pipeline: str):
    """TLV L-amygdala <-> vmPFC FC at ses-1 and ses-3 for 'anatomical' (no GSR) or 'global' (GSR).
    Scan rule = kpe_config.scan_drop() (scrubbed > 115, retained < 300 dropped; rest2 wins), minus EXCLUDE.
    Returns DataFrame[subject, arm, fc_ses1, fc_ses3, delta_fc, improvement] (improvement = -dCAPS) and a
    counts dict."""
    d = cfg.DATASETS["tau"]
    norm = cs.subject_id
    drop = cfg.scan_drop(d[f"{pipeline}_scrub"], normalize=norm)
    chosen = {}
    for tf in sorted(cfg.find_ts_files(d[pipeline], "tian_s2")):
        b = os.path.basename(str(tf))
        sub = norm(re_search(r"sub-[A-Za-z0-9]+", b)); ses = re_search(r"ses-[A-Za-z0-9]+", b)
        task = re_search(r"task-[A-Za-z0-9]+", b)
        if (sub, ses) in drop or sub in EXCLUDE:
            continue
        if (sub, ses) not in chosen or task > chosen[(sub, ses)][1]:
            chosen[(sub, ses)] = (tf, task)
    fc = {}
    for key, (tf, _) in chosen.items():
        ho = pd.read_csv(str(tf).replace("tian_s2", "harvard_oxford_cortical"))
        ho.columns = [str(c).replace("\n", " ").strip() for c in ho.columns]
        fc[key] = parcel_edge_fc(pd.read_csv(tf), ho)
    arms, imp = tlv_arms(), improvement()
    rows = [dict(subject=s, arm=arms[s], fc_ses1=fc[(s, "ses-1")], fc_ses3=fc[(s, "ses-3")],
                 delta_fc=fc[(s, "ses-3")] - fc[(s, "ses-1")], improvement=imp[s])
            for s in sorted({k[0] for k in fc})
            if (s, "ses-1") in fc and (s, "ses-3") in fc and s in imp and s in arms]
    df = pd.DataFrame(rows)
    both = {k[0] for k in fc if (k[0], "ses-1") in fc and (k[0], "ses-3") in fc}
    no_pair = sorted(s for s in arms if s in imp and s not in EXCLUDE and s not in both)
    return df, dict(pipeline=pipeline, scans_used=len(chosen), n=len(df), per_arm=df.arm.value_counts().to_dict(),
                    arm_caps_no_ses1_ses3_pair=no_pair)


def re_search(pattern: str, text: str) -> str:
    import re
    return re.search(pattern, text).group(0)


def add_common_args(ap):
    """--out, --check/--no-check, --versions (comma list; default 4mm,6mm; use 4mm without the 6 mm data)."""
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--versions", default="4mm,6mm", help="comma list of 4mm,6mm (default both)")


def parse_versions(s: str) -> tuple:
    v = tuple(x.strip() for x in s.split(",") if x.strip())
    assert v and set(v) <= {"4mm", "6mm"}, f"--versions must be 4mm and/or 6mm, got {s!r}"
    return v


def pooled_run(pair_name: str, pair: tuple, versions=("4mm", "6mm")):
    """Build tables for one pair on each version; returns (per_subject, results rows, info).
    Results rows: pooled centred test + per-arm Spearman/Pearson, per version."""
    subj, rows, info = [], [], {}
    for v in versions:
        df, inf = build_subject_table(v, {pair_name: pair})
        info[v] = inf
        subj.append(df)
        p = pooled_centered(df)
        rows.append(dict(version=v, pair=pair_name, arm="POOLED (centred within arm)", **p))
        for r in per_arm_stats(df):
            rows.append(dict(version=v, pair=pair_name, **r))
    return pd.concat(subj), pd.DataFrame(rows), info


def print_scan_counts(info: dict):
    """Short scans used / dropped per version and arm (expected 4 mm 10/10/10, 6 mm 10/10/11)."""
    for v, i in info.items():
        print(f"[{v}] scans used {i['used']}, per arm {i['per_arm']}, dropped (motion/short) {i['dropped_motion']}, "
              f"missing files {i['missing_files']}, arm+CAPS but no ses-1 scan {i['arm_caps_no_scan']}")


def check_range(label: str, val: float, lo: float, hi: float, poster) -> bool:
    """Monte-Carlo permutation p-values wobble with the seed: accept lo <= val <= hi around the poster value."""
    good = lo <= float(val) <= hi
    print(f"  [{'OK  ' if good else 'FAIL'}] {label}: got {float(val):.5f}, poster {poster} (accepted {lo}-{hi})")
    return good
