"""Result 4 (static FC, not dFC): change in left-amygdala <-> vmPFC FC (day 30 - day 0) vs CAPS improvement.

Poster numbers: GSR n=28 rho=+.52 p=.0045; no-GSR rho=+.41 p=.032 (improvement units).
Also prints the checker's diagnostic (baseline FC, day-30 FC, arm-centred rho).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import dfc_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)   # --versions is accepted but unused: result 4 is 4 mm static FC only
    a = ap.parse_args()
    out = a.out or C.default_out("dfc_poster_amygdala_vmpfc_change")

    res, subj, info = [], [], {}
    for pipe, label in [("global", "GSR"), ("anatomical", "noGSR")]:
        df, cnt = C.static_fc_table(pipe)
        info[label] = cnt
        subj.append(df.assign(pipeline=label))
        row = lambda test, x, y, **kw: res.append(dict(pipeline=label, test=test, n=len(df), **kw,
                    **dict(zip(["rho", "p"], stats.spearmanr(x, y))),
                    pearson_r=stats.pearsonr(x, y)[0], pearson_p=stats.pearsonr(x, y)[1]))
        row("delta_fc (day30-day0) vs improvement  [MAIN]", df.delta_fc, df.improvement)
        row("baseline FC (ses-1) vs improvement  [diagnostic]", df.fc_ses1, df.improvement)
        row("day-30 FC (ses-3) vs improvement  [diagnostic]", df.fc_ses3, df.improvement)
        cv = C.center_within_arm(df.delta_fc, df.arm); cy = C.center_within_arm(df.improvement, df.arm)
        row("delta_fc vs improvement, centred within arm  [diagnostic]", cv, cy)
        for arm in C.ARMS:
            g = df[df.arm == arm]
            row(f"delta_fc vs improvement, {C.ARM_LABEL[arm]} only", g.delta_fc, g.improvement)
            res[-1]["n"] = len(g)
    R = pd.DataFrame(res)
    S = pd.concat(subj)
    pd.set_option("display.width", 200)
    print("Scans/subjects:", info)
    print(R[["pipeline", "test", "n", "rho", "p"]].round(4).to_string(index=False))

    methods = C.common_methods("") .split("\n")[0] + f"""
RESULT 4 (static FC): left amygdala (Tian S2 lAMY-lh + mAMY-lh) <-> Harvard-Oxford 'Frontal Medial Cortex'.
Parcel-edge FC: Pearson per parcel pair (2 pairs) -> Fisher z -> mean. delta_fc = FC(ses-3) - FC(ses-1).
Outcome: CAPS improvement = baseline - day 30 (positive = better), via caps_source.
Scan rule here follows kpe_config.scan_drop() (scrubbed > {C.SCRUB_MAX} dropped; retained < {C.cfg.MIN_RETAINED_VOLUMES} dropped; rest2 wins), minus {sorted(C.EXCLUDE)}.
Pipelines: global = GSR (poster panel), anatomical = no GSR. Spearman primary; Pearson alongside. Nominal p, no FDR; post hoc.
Diagnostic rows (baseline FC, day-30 FC, arm-centred) are for transparency: the change score largely reflects baseline connectivity.
Counts: {info}
"""
    C.finish(out, S, R, methods)
    print("OUT", out)

    if a.check:
        g = lambda lab, t: R[(R.pipeline == lab) & (R.test.str.startswith(t))].iloc[0]
        m_g, m_n = g("GSR", "delta_fc (day30"), g("noGSR", "delta_fc (day30")
        ok = C.run_checks([("GSR n", m_g.n, 28, 0), ("GSR rho", m_g.rho, .52, 2), ("GSR p", m_g.p, .0045, 4),
                           ("noGSR n", m_n.n, 28, 0), ("noGSR rho", m_n.rho, .41, 2), ("noGSR p", m_n.p, .032, 3),
                           ("GSR baseline rho", g("GSR", "baseline").rho, -.53, 2),
                           ("GSR baseline p", g("GSR", "baseline").p, .0035, 4),
                           ("GSR day-30 rho", g("GSR", "day-30").rho, .01, 2),
                           ("GSR centred rho", g("GSR", "delta_fc vs improvement, centred").rho, .25, 2),
                           ("GSR centred p", g("GSR", "delta_fc vs improvement, centred").p, .21, 2)])
        print("CHECK", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
