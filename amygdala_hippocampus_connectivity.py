#!/usr/bin/env python3
"""
amygdala_hippocampus_connectivity.py
====================================
Resting-state amygdala <-> hippocampus functional connectivity (both from Tian S2),
grounded in Or Duek et al. 2023 (ketamine reduced amygdala <-> posterior-hippocampus FC).

Two result sets, produced from ONE computation:
  (A) OR DUEK PRIMARY  -- a small pre-specified family: amygdala (whole / L / R)
      -> POSTERIOR hippocampus, ketamine vs placebo. This is the "did we replicate them?"
      test, named in advance.
  (B) FULL GRID        -- every amygdala option x every hippocampus option:
        amygdala : AMY(whole), AMY-R, AMY-L, AMY-lateral, AMY-medial, + 4 raw parcels
        hippo    : HIP(whole), HIP-R, HIP-L, HIP-anterior, HIP-posterior, + 4 raw parcels
      = 9 x 9 = 81 FC pairs, x 2 pipelines x 3 session contrasts x arm/site.

FC  : Pearson r between the two ROI mean time-series, Fisher-z (arctanh).
MOTION: scrubbed_volumes > 115 dropped AND scans with retained volumes (total_vols - scrubbed_volumes) < 300 dropped (>=5 min clean at TR=1; shared rule kpe_config.scan_drop; length judged on the analysed run, rest2 wins).
CHANGE: within-subject delta = FC(follow) - FC(base).
GROUPS: Welch t on the deltas + pooled Cohen's d; BH-FDR across the 9 hippo targets
        within each (site, pipeline, contrast, comparison, amygdala seed).
DOSE : ketamine-vs-placebo run per-site AND combined; the 3 dose contrasts (ket0.5-vs-pbo,
        ket0.2-vs-pbo, ket0.5-vs-ket0.2) run on COMBINED only.
CONSISTENCY: an effect flagged consistent_pipeline when the SAME (site,seed,target,
        comparison,contrast) is nominally p<.05 in BOTH anatomical and global, same sign.

SESSION MAP (assumption, flagged): ses-1 = baseline, ses-2 = end of treatment (~day 7),
        ses-3 = 30-day follow-up. Or Duek's FC effect was at end of treatment (ses-1->ses-2),
        transient by 30d; ses-1->ses-3 is the long-term check.
"""
from __future__ import annotations
import os, re
from datetime import datetime
import numpy as np
import pandas as pd
import math
try:
    from scipy.stats import ttest_ind as _scipy_ttest_ind
    def welch_ttest(a, b): return _scipy_ttest_ind(a, b, equal_var=False)
except Exception:
    def _betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0; d = 1.0 - qab * x / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0 / d; h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < EPS: break
        return h
    def _betai(a, b, x):
        if x <= 0.0: return 0.0
        if x >= 1.0: return 1.0
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * _betacf(a, b, x) / a
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    def welch_ttest(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        n1, n2 = len(a), len(b); v1, v2 = a.var(ddof=1), b.var(ddof=1)
        se2 = v1 / n1 + v2 / n2
        if se2 <= 0: return (0.0, 1.0)
        t = (a.mean() - b.mean()) / math.sqrt(se2)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        p = _betai(df / 2.0, 0.5, df / (df + t * t))
        return (t, p)

import kpe_config as cfg

DATASETS_TO_RUN = ["tau", "yale", "combined"]
PIPELINES       = ["anatomical", "global"]
CONTRASTS       = [("ses-1", "ses-2"), ("ses-1", "ses-3"), ("ses-2", "ses-3")]
SCRUB_THRESHOLD = cfg.SCRUB_THRESHOLD   # shared rule: kpe_config.py (also MIN_RETAINED_VOLUMES)
FISHER_Z        = True
MIN_PER_GROUP   = 2

AMYG_SIGNALS = {
    "AMY":         ["lAMY-lh", "mAMY-lh", "lAMY-rh", "mAMY-rh"],
    "AMY-R":       ["lAMY-rh", "mAMY-rh"],
    "AMY-L":       ["lAMY-lh", "mAMY-lh"],
    "AMY-lateral": ["lAMY-rh", "lAMY-lh"],
    "AMY-medial":  ["mAMY-rh", "mAMY-lh"],
    "lAMY-rh": ["lAMY-rh"], "mAMY-rh": ["mAMY-rh"],
    "lAMY-lh": ["lAMY-lh"], "mAMY-lh": ["mAMY-lh"],
}
HIPPO_SIGNALS = {
    "HIP":           ["aHIP-lh", "pHIP-lh", "aHIP-rh", "pHIP-rh"],
    "HIP-R":         ["aHIP-rh", "pHIP-rh"],
    "HIP-L":         ["aHIP-lh", "pHIP-lh"],
    "HIP-anterior":  ["aHIP-rh", "aHIP-lh"],
    "HIP-posterior": ["pHIP-rh", "pHIP-lh"],
    "aHIP-rh": ["aHIP-rh"], "pHIP-rh": ["pHIP-rh"],
    "aHIP-lh": ["aHIP-lh"], "pHIP-lh": ["pHIP-lh"],
}
PRIMARY_SEEDS   = {"AMY", "AMY-R", "AMY-L"}
PRIMARY_TARGETS = {"HIP-posterior", "pHIP-rh", "pHIP-lh"}

COMPARISONS = {
    "ketamine_vs_placebo": ({"ket0.5", "ket0.2"}, {"placebo"}),
    "ket0.5_vs_placebo":   ({"ket0.5"},           {"placebo"}),
    "ket0.2_vs_placebo":   ({"ket0.2"},           {"placebo"}),
    "ket0.5_vs_ket0.2":    ({"ket0.5"},           {"ket0.2"}),
}

def normalize_subject_id(value) -> str:
    value = str(value).strip()
    if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
        digits = re.findall(r"\d+", value)
        return f"sub-{int(digits[0]):03d}" if digits else value
    m = re.match(r".*?(\d+)", value)
    return f"sub-{int(m.group(1)):03d}" if m else value

def _norm_group(v): return str(v).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
def treatment_to_name(v):
    v = _norm_group(v)
    if v == "a": return "ket0.5"
    if v == "b": return "ket0.2"
    if v == "c": return "placebo"
    if v in ("ket0.5","ketamine0.5","ketamine05","ket05"): return "ket0.5"
    if v in ("ket0.2","ketamine0.2","ketamine02","ket02"): return "ket0.2"
    if v in ("placebo","midazolam","control"): return "placebo"
    return None

def load_treatments(rand_path_or_list) -> dict:
    """subject -> arm, from the ONE source (caps_source.py → behavioural_yale_tlv.csv).
    Raises if a scanned subject has no arm. Never reads randomization files directly."""
    import caps_source as CS
    return CS.arms_for(rand_path_or_list)

def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = p.size
    order = np.argsort(p); ranked = p[order]*n/(np.arange(n)+1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked,0,1); return out

def cohen_d(x, y):
    x,y = np.asarray(x,float), np.asarray(y,float); nx,ny = len(x),len(y)
    if nx < 2 or ny < 2: return np.nan
    sp = np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
    return (x.mean()-y.mean())/sp if sp > 0 else np.nan

def _entities(path):
    b = os.path.basename(path)
    return (re.search(r"sub-[A-Za-z0-9]+",b), re.search(r"ses-[A-Za-z0-9]+",b),
            re.search(r"task-[A-Za-z0-9]+",b), re.search(r"run-[A-Za-z0-9]+",b))
def _scan_key(path):
    s = _entities(path); return tuple(m.group(0) if m else None for m in s)

def build_scrub_drop(scrub_path_or_list):
    """Shared motion + minimum-length rule: kpe_config.scan_drop()."""
    return cfg.scan_drop(scrub_path_or_list, normalize=normalize_subject_id)

def _sig_series(df, sigdef):
    out = {}
    for name, members in sigdef.items():
        present = [m for m in members if m in df.columns]
        if present: out[name] = df[present].mean(axis=1)
    return out

def compute_fc(ts_dirs, scrub_drop):
    tian_files = cfg.find_ts_files(ts_dirs, "tian_s2")
    fc = {}; n = 0; ndrop = 0
    for tf in tian_files:
        sub, ses = _scan_key(tf)[0], _scan_key(tf)[1]
        if (normalize_subject_id(sub), ses) in scrub_drop: ndrop += 1; continue
        tian = pd.read_csv(tf)
        amy = _sig_series(tian, AMYG_SIGNALS); hip = _sig_series(tian, HIPPO_SIGNALS)
        d = {}
        for an, asig in amy.items():
            av = asig.values
            for hn, hsig in hip.items():
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.corrcoef(av, hsig.values)[0,1]
                if FISHER_Z and np.isfinite(r):
                    r = np.arctanh(min(max(r,-0.9999),0.9999))
                d[(an,hn)] = r
        fc[(normalize_subject_id(sub), ses)] = d; n += 1
    print(f"    scans used={n}  dropped_motion={ndrop}")
    return fc

def run_dataset(dataset):
    print(f"\n{'='*70}\nDATASET: {dataset}\n{'='*70}")
    treatments = load_treatments(cfg.DATASETS[dataset]["randomization"])
    comps = COMPARISONS if dataset == "combined" else {"ketamine_vs_placebo": COMPARISONS["ketamine_vs_placebo"]}
    rows = []
    for pipeline in PIPELINES:
        print(f"  pipeline: {pipeline}")
        drop = build_scrub_drop(cfg.DATASETS[dataset].get(f"{pipeline}_scrub"))
        fc = compute_fc(cfg.DATASETS[dataset][pipeline], drop)
        subjects = {s for (s,_) in fc}
        for (base, follow) in CONTRASTS:
            for comp_name, (g1, g2) in comps.items():
                for aname in AMYG_SIGNALS:
                    batch = []
                    for hname in HIPPO_SIGNALS:
                        d1, d2 = [], []
                        for subj in subjects:
                            b = fc.get((subj,base), {}).get((aname,hname))
                            f = fc.get((subj,follow), {}).get((aname,hname))
                            if b is None or f is None or not (np.isfinite(b) and np.isfinite(f)): continue
                            dv = float(f)-float(b); tr = treatments.get(subj)
                            if tr in g1: d1.append(dv)
                            elif tr in g2: d2.append(dv)
                        if len(d1) >= MIN_PER_GROUP and len(d2) >= MIN_PER_GROUP:
                            t, p = welch_ttest(d1, d2)
                            batch.append((hname,t,p,cohen_d(d1,d2),np.mean(d1)-np.mean(d2),len(d1),len(d2)))
                    if not batch: continue
                    fdr = bh_fdr(np.array([b[2] for b in batch]))
                    for (hname,t,p,dd,md,n1,n2), pf in zip(batch,fdr):
                        rows.append({
                            "family":"amygdala<->hippocampus","site":dataset,"pipeline":pipeline,
                            "seed":aname,"region":hname,"comparison":comp_name,
                            "sessions":f"{base}_to_{follow}","n_group1":n1,"n_group2":n2,
                            "mean_diff_g1_minus_g2":md,"t":t,"p":p,"cohens_d":dd,"p_fdr":pf,
                            "sig05":p<0.05,"sigFDR":pf<0.05,
                            "or_duek_primary":(comp_name=="ketamine_vs_placebo"
                                               and aname in PRIMARY_SEEDS and hname in PRIMARY_TARGETS),
                        })
    return pd.DataFrame(rows)

def add_consistency(df):
    df = df.copy()
    df["signdir"] = np.sign(df["mean_diff_g1_minus_g2"])
    key = ["site","seed","region","comparison","sessions"]
    def _c(g):
        gg = g[g.p < 0.05]
        return bool(set(gg.pipeline) >= {"anatomical","global"} and gg.signdir.nunique()==1 and len(gg)>=2)
    flags = df.groupby(key).apply(_c).rename("consistent_pipeline").reset_index()
    return df.merge(flags, on=key, how="left").drop(columns=["signdir"])

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.OUTPUT_DATA_DIR / f"amygdala_hippocampus_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [run_dataset(ds) for ds in DATASETS_TO_RUN]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = add_consistency(df).sort_values("p", na_position="last").reset_index(drop=True)

    df.to_csv(out_dir / "amyg_hip_all_tests.csv", index=False)
    prim = df[df.or_duek_primary].sort_values("p").reset_index(drop=True)
    prim.to_csv(out_dir / "amyg_hip_or_duek_primary.csv", index=False)
    cons = df[(df.sig05) & (df.consistent_pipeline)].sort_values(["comparison","p"]).reset_index(drop=True)
    cons.to_csv(out_dir / "amyg_hip_consistent.csv", index=False)

    (out_dir / "METHODS.txt").write_text(__doc__ + f"\nRUN: {ts}\nTOTAL TESTS: {len(df)}\n")

    print(f"\n{'='*70}\nSUMMARY  (out: {out_dir})\n{'='*70}")
    print(f"total tests: {len(df):,}")
    for site, g in df.groupby("site"):
        print(f"  {site:9} tests={len(g):>5} | p<.05={int(g.sig05.sum()):>4} | FDR={int(g.sigFDR.sum()):>3} | consistent={int(g.consistent_pipeline.sum()):>3}")

    print(f"\n--- (A) OR DUEK PRIMARY: amygdala(whole/L/R) -> posterior hippocampus, ketamine vs placebo ---")
    print(f"    {len(prim)} tests; direction we expect = NEGATIVE mean_diff (ketamine reduces FC)")
    for r in prim.itertuples():
        star = "**" if r.sigFDR else ("* " if r.sig05 else "  ")
        print(f"  {star}{r.site:8} {r.pipeline:10} {r.seed:6}->{r.region:13} {r.sessions:14} "
              f"n={r.n_group1}/{r.n_group2} d={r.cohens_d:+.2f} p={r.p:.4f} pFDR={r.p_fdr:.3f}")

    print(f"\n--- (B) FULL GRID: strongest consistent (both-pipeline) effects, top 20 by p ---")
    top = cons.head(20)
    if top.empty: print("   none survived the both-pipeline consistency filter.")
    for r in top.itertuples():
        star = "**" if r.sigFDR else "* "
        print(f"  {star}{r.site:8} {r.pipeline:10} {r.seed:8}->{r.region:13} {r.comparison:20} "
              f"{r.sessions:14} d={r.cohens_d:+.2f} p={r.p:.4f} pFDR={r.p_fdr:.3f}")

    print(f"\nwrote: amyg_hip_all_tests.csv ({len(df)}), amyg_hip_or_duek_primary.csv ({len(prim)}), "
          f"amyg_hip_consistent.csv ({len(cons)})")

if __name__ == "__main__":
    main()
