"""Result 3: right amygdala (Tian lAMY-rh + mAMY-rh) <-> Control dFC variance vs CAPS improvement.
Poster: 4 mm pooled (within-arm centred) n=30 rho=-.56, permutation p~.002; 6 mm rho=-.55; per arm."""
import argparse, sys
from pathlib import Path
import dfc_common as C

PAIR = "AmygR-Control"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    a = ap.parse_args()
    S, R, info = C.pooled_run(PAIR, ("AmygR", "Control"), C.parse_versions(a.versions))
    C.print_scan_counts(info)
    print(R[["version", "arm", "n", "rho_centred", "p_perm", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]].round(4).to_string(index=False))
    out = a.out or C.default_out("dfc_poster_ramygdala_control")
    C.finish(out, S, R, C.common_methods(f"RESULT 3: right amygdala (lAMY-rh + mAMY-rh) <-> Schaefer Control. Counts: {info}\n", n_perm=int(R.n_perm.dropna().iloc[0])))
    print("OUT", out)
    if a.check:
        assert "4mm" in a.versions, "--check needs 4mm; use --no-check otherwise"
        p4 = R[(R.version == "4mm") & R.arm.str.startswith("POOLED")].iloc[0]
        chk = [("4mm pooled n", p4.n, 30, 0), ("4mm pooled rho", p4.rho_centred, -.56, 2)]
        if "6mm" in a.versions:
            p6 = R[(R.version == "6mm") & R.arm.str.startswith("POOLED")].iloc[0]
            chk.append(("6mm pooled rho", p6.rho_centred, -.55, 2))
        ok = C.run_checks(chk)
        ok &= C.check_range("4mm permutation p", p4.p_perm, 0.0018, 0.0030, "~.002")
        print("n_perm used:", int(p4.n_perm))
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
