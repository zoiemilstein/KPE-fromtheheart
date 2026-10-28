"""Result 5: Salience <-> DMN dFC variance vs CAPS improvement, per arm (TLV, baseline).
Poster (6 mm March extraction, Pearson): ket0.5 n=10 r=-.73 p=.017; ket0.2 n=10 r=-.54 p=.111; midazolam n=11 r=-.25 p=.452;
interaction ket0.5 vs midazolam p=.044, vs ket0.2 p=.035. 4 mm reported for transparency (ket0.5 r=-.58 p=.079)."""
import argparse, sys
from pathlib import Path
import pandas as pd
import dfc_common as C

PAIR = "Salience-DMN"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    a = ap.parse_args()
    subj, rows, info = [], [], {}
    for v in tuple(reversed(C.parse_versions(a.versions))):
        df, inf = C.build_subject_table(v, {PAIR: ("Salience", "DMN")})
        info[v] = inf; subj.append(df)
        for r in C.per_arm_stats(df):
            rows.append(dict(version=v, pair=PAIR, test="per arm", **r))
        for b in ("placebo", "ket0.2"):
            i = C.interaction(df, "ket0.5", b)
            rows.append(dict(version=v, pair=PAIR, test="interaction (OLS improvement ~ arm*var)", arm=i["contrast"],
                             n=i["n"], coef=i["coef"], p_interaction=i["p"]))
    S, R = pd.concat(subj), pd.DataFrame(rows)
    C.print_scan_counts(info)
    print(R[["version", "test", "arm", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "coef", "p_interaction"]].round(4).to_string(index=False))
    out = a.out or C.default_out("dfc_poster_sn_dmn_ketamine")
    C.finish(out, S, R, C.common_methods(
        f"RESULT 5: Salience <-> DMN. Primary statistic Pearson (Spearman reported alongside). 6 mm = March extraction (poster); 4 mm for transparency.\n"
        f"Interaction: OLS improvement ~ arm*var on two arms (columns coef, p_interaction). No multiplicity correction. Counts: {info}\n"))
    print("OUT", out)
    if a.check:
        g = lambda v, arm: R[(R.version == v) & (R.test == "per arm") & (R.arm == arm)].iloc[0]
        it = lambda v, c: R[(R.version == v) & R.test.str.startswith("interaction") & (R.arm == c)].iloc[0]
        chk = []
        if "6mm" in a.versions:
            k, k2, m = g("6mm", "ket0.5"), g("6mm", "ket0.2"), g("6mm", "midazolam")
            chk += [("6mm ket0.5 n", k.n, 10, 0), ("6mm ket0.5 r", k.pearson_r, -.73, 2), ("6mm ket0.5 p", k.pearson_p, .017, 3),
                    ("6mm ket0.2 n", k2.n, 10, 0), ("6mm ket0.2 r", k2.pearson_r, -.54, 2), ("6mm ket0.2 p", k2.pearson_p, .111, 3),
                    ("6mm midazolam n", m.n, 11, 0), ("6mm midazolam r", m.pearson_r, -.25, 2), ("6mm midazolam p", m.pearson_p, .452, 3),
                    ("6mm interaction vs midazolam p", it("6mm", "ket0.5 vs midazolam").p_interaction, .044, 3),
                    ("6mm interaction vs ket0.2 p", it("6mm", "ket0.5 vs ket0.2").p_interaction, .035, 3),
                    ("6mm ket0.5 Spearman p (~.20)", k.spearman_p, .20, 2)]
        if "4mm" in a.versions:
            chk += [("4mm ket0.5 r", g("4mm", "ket0.5").pearson_r, -.58, 2), ("4mm ket0.5 p", g("4mm", "ket0.5").pearson_p, .079, 3),
                    ("4mm midazolam Spearman rho", g("4mm", "midazolam").spearman_rho, -.70, 2),
                    ("4mm midazolam Spearman p", g("4mm", "midazolam").spearman_p, .024, 3)]
        assert chk, "nothing to check"
        ok = C.run_checks(chk)
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
