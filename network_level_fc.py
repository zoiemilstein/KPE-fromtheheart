#!/usr/bin/env python3
"""
network_level_fc.py — limbic→network and network×network FC on the 2026-08 timeseries.

Replaces the June/July runs (tian_limbic_subset_to_networks.py, schaefer_network_connectivity.py),
which read the old extraction. Follows the KPE methods conventions (CLAUDE.md):

  • Data: timeseries_2026-08/{tlv,yale}/{anatomical,global} via kpe_config.DATASETS (tau / yale / combined).
  • FC by PARCEL-EDGE AVERAGING: Pearson r for every parcel pair → Fisher-z → mean over the pairs
    that make up a seed→network or network×network edge. Timeseries are never averaged first.
      - limbic→network: every Tian S2 member parcel of the seed × every Schaefer-400 parcel of the network.
      - network×network: every parcel of net A × every parcel of net B; within-network = all distinct pairs.
  • Schaefer columns RE-NAMED with the official parcel list (schaefer_relabel): the extracted CSVs
    have names shifted by one ('Background' = parcel 1). See atlas/ and CLAUDE.md.
  • Motion: scrubbed_volumes > 115 dropped AND scans with retained volumes (total_vols - scrubbed_volumes) < 300 dropped (>=5 min clean at TR=1; shared rule kpe_config.scan_drop; length judged on the analysed run, rest2 wins).
  • Δ = FC(follow-up) − FC(ses-1); Welch t + pooled Cohen's d; min 2 per group.
  • Comparisons: per site ketamine vs placebo; combined adds 0.5 vs placebo, 0.2 vs placebo, 0.5 vs 0.2.
  • BH-FDR within each family = (analysis, site, pipeline, comparison, sessions).
  • Sessions: ses-1 = day 0, ses-2 = day 7, ses-3 = day 30.

Writes a NEW timestamped folder in output_data/; touches nothing else.
"""
import os, itertools
from datetime import datetime
import numpy as np, pandas as pd
import kpe_config as cfg, amygdala_hippocampus_connectivity as A

LIMBIC_SEEDS = {
    "amygdala":              ["lAMY-lh", "mAMY-lh", "lAMY-rh", "mAMY-rh"],
    "ant_hippocampus":       ["aHIP-lh", "aHIP-rh"],
    "post_hippocampus":      ["pHIP-lh", "pHIP-rh"],
    "anteromedial_thalamus": ["THA-DA-lh", "THA-VA-lh", "THA-DA-rh", "THA-VA-rh"],
    "nucleus_accumbens":     ["NAc-shell-lh", "NAc-core-lh", "NAc-shell-rh", "NAc-core-rh"],
}
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_LABEL = {"Cont": "Control", "Default": "DMN"}
CONTRASTS = [("ses-1", "ses-2"), ("ses-1", "ses-3")]
DUPLICATES = set()


OFFICIAL = [l.split("\t")[1] for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "atlas", "Schaefer2018_400Parcels_7Networks_order.txt")) if l.strip()]


def schaefer_relabel(df):
    """The 2026-08 Schaefer CSVs carry names shifted by one: nilearn's label list starts with
    'Background', so data column k (parcel k+1) was named after parcel k. Verified 2026-10-19:
    names = ['Background'] + official[:399], and the shifted reading gives more coherent
    networks in 66/66 scans. Re-name every data column with the official parcel it holds."""
    cols = list(df.columns)
    if cols == ["Background"] + OFFICIAL[:399]:
        df = df.copy(); df.columns = OFFICIAL; return df
    if cols == OFFICIAL:
        return df
    raise ValueError("unexpected Schaefer column layout — check before using")


def net_of(col):
    return col.split("_")[2]


def scan_fc(tian_path, sch_path):
    """One scan → {('limbic', seed, net): z, ('network', netA, netB): z}."""
    tian = pd.read_csv(tian_path)
    sch = schaefer_relabel(pd.read_csv(sch_path))
    seed_cols = [c for cols in LIMBIC_SEEDS.values() for c in cols if c in tian.columns]
    X = np.column_stack([tian[seed_cols].values, sch.values])
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X, rowvar=False)
    Z = np.arctanh(np.clip(R, -0.9999, 0.9999))
    k = len(seed_cols)
    idx_seed = {c: i for i, c in enumerate(seed_cols)}
    net_idx = {n: [k + j for j, c in enumerate(sch.columns) if net_of(c) == n] for n in NETWORKS}
    out = {}
    for seed, members in LIMBIC_SEEDS.items():
        si = [idx_seed[c] for c in members if c in idx_seed]
        if not si:
            continue
        for n in NETWORKS:
            out[("limbic", seed, n)] = float(np.nanmean(Z[np.ix_(si, net_idx[n])]))
    for a, b in itertools.combinations_with_replacement(NETWORKS, 2):
        block = Z[np.ix_(net_idx[a], net_idx[b])]
        if a == b:
            block = block[np.triu_indices_from(block, k=1)]
        out[("network", a, b)] = float(np.nanmean(block))
    return out


def collect(dataset, pipeline):
    roots = cfg.DATASETS[dataset][pipeline]
    drop = A.build_scrub_drop(cfg.DATASETS[dataset].get(f"{pipeline}_scrub"))
    sch_files = {A._scan_key(str(f)): f for f in cfg.find_ts_files(roots, "schaefer400")}
    fc, ndrop, nmiss = {}, 0, 0
    for tf in cfg.find_ts_files(roots, "tian_s2"):
        key = A._scan_key(str(tf))
        sub, ses = A.normalize_subject_id(key[0]), key[1]
        if (sub, ses) in drop:
            ndrop += 1
            continue
        sf = sch_files.get(key)
        if sf is None:
            nmiss += 1
            continue
        if (sub, ses) in fc:   # same rule as amygdala_hippocampus_connectivity.compute_fc: later file (rest2) wins
            DUPLICATES.add((dataset, pipeline, sub, ses))
        fc[(sub, ses)] = scan_fc(tf, sf)
    print(f"  {dataset}/{pipeline}: scans used={len(fc)}  dropped_motion={ndrop}  no_schaefer_match={nmiss}")
    return fc


def run():
    rows = []
    for dataset in ["tau", "yale", "combined"]:
        treat = A.load_treatments(cfg.DATASETS[dataset]["randomization"])
        comps = A.COMPARISONS if dataset == "combined" else {"ketamine_vs_placebo": A.COMPARISONS["ketamine_vs_placebo"]}
        for pipeline in ["anatomical", "global"]:
            fc = collect(dataset, pipeline)
            subjects = {s for (s, _) in fc}
            edges = sorted({e for d in fc.values() for e in d})
            for base, follow in CONTRASTS:
                for comp, (g1, g2) in comps.items():
                    for analysis in ["limbic", "network"]:
                        batch = []
                        for e in [e for e in edges if e[0] == analysis]:
                            d1, d2 = [], []
                            for s in subjects:
                                b, f = fc.get((s, base), {}).get(e), fc.get((s, follow), {}).get(e)
                                if b is None or f is None or not (np.isfinite(b) and np.isfinite(f)):
                                    continue
                                tr = treat.get(s)
                                (d1 if tr in g1 else d2 if tr in g2 else []).append(f - b)
                            if len(d1) >= A.MIN_PER_GROUP and len(d2) >= A.MIN_PER_GROUP:
                                t, p = A.welch_ttest(d1, d2)
                                batch.append((e, len(d1), len(d2), np.mean(d1) - np.mean(d2), t, p, A.cohen_d(d1, d2)))
                        if not batch:
                            continue
                        fdr = A.bh_fdr([b[5] for b in batch])
                        for (e, n1, n2, md, t, p, dd), q in zip(batch, fdr):
                            rows.append({
                                "family": "limbic→network" if analysis == "limbic" else "network×network",
                                "site": dataset, "pipeline": pipeline,
                                "seed": e[1] if analysis == "limbic" else NET_LABEL.get(e[1], e[1]),
                                "region": NET_LABEL.get(e[2], e[2]),
                                "comparison": comp, "sessions": f"{base}_to_{follow}",
                                "n_group1": n1, "n_group2": n2, "mean_diff_g1_minus_g2": md,
                                "t": t, "p": p, "cohens_d": dd, "p_fdr": q,
                                "sig05": p < 0.05, "sigFDR": q < 0.05,
                            })
    return pd.DataFrame(rows)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cfg.OUTPUT_DATA_DIR / f"network_level_fc_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    df = run()
    for fam, fname in [("limbic→network", "limbic_network_all_tests.csv"),
                       ("network×network", "network_network_all_tests.csv")]:
        d = df[df.family == fam].sort_values("p")
        d.to_csv(out / fname, index=False)
        print(f"{fam}: {len(d)} tests · p<.05: {int(d.sig05.sum())} · FDR<.05: {int(d.sigFDR.sum())}")
    dup = sorted({(s, ses) for (_, _, s, ses) in DUPLICATES})
    (out / "METHODS.txt").write_text(__doc__ + f"\nScans with both rest1 and rest2 (rest2 used, as in the other analyses): {dup}\n")
    print("rest1+rest2 duplicates (rest2 used):", dup)
    print(f"out: {out}")


if __name__ == "__main__":
    main()
