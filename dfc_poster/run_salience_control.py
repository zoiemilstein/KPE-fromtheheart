"""Result 2: Salience <-> Control dFC variance vs CAPS improvement (TLV, baseline).
Poster: 4 mm, all arms pooled (within-arm centred) n=30 rho=-.66, permutation p~.0002. Also 6 mm pooled."""
import argparse, sys
from pathlib import Path
import dfc_common as C

PAIR = "Salience-Control"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    a = ap.parse_args()
    S, R, info = C.pooled_run(PAIR, ("Salience", "Control"), C.parse_versions(a.versions))
    C.print_scan_counts(info)
    print(R[["version", "arm", "n", "rho_centred", "p_perm", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]].round(4).to_string(index=False))
    out = a.out or C.default_out("dfc_poster_salience_control")
    C.finish(out, S, R, C.common_methods(f"RESULT 2: {PAIR}. Per-arm rows: Spearman + Pearson. Counts: {info}\n", n_perm=int(R.n_perm.dropna().iloc[0])))
    print("OUT", out)
    if a.check:
        assert "4mm" in a.versions, "--check needs 4mm; use --no-check otherwise"
        p4 = R[(R.version == "4mm") & R.arm.str.startswith("POOLED")].iloc[0]
        arm4 = R[(R.version == "4mm") & ~R.arm.str.startswith("POOLED")].set_index("arm")
        ok = C.run_checks([("4mm pooled n", p4.n, 30, 0), ("4mm pooled rho", p4.rho_centred, -.66, 2),
                           ("4mm ket0.5 rho", arm4.loc["ket0.5", "spearman_rho"], -.63, 2),
                           ("4mm ket0.2 rho", arm4.loc["ket0.2", "spearman_rho"], -.54, 2),
                           ("4mm midazolam rho", arm4.loc["midazolam", "spearman_rho"], -.88, 2)])
        ok &= C.check_range("4mm permutation p", p4.p_perm, 0.0001, 0.0003, "~.0002")
        print("n_perm used:", int(p4.n_perm))
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
