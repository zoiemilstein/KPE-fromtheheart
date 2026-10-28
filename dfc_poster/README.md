# dFC-variance poster analyses (KPE, ketamine + prolonged exposure for PTSD)

Code behind four poster results, TLV site, resting-state fMRI, no-GSR ("anatomical") timeseries unless stated.
All analyses are **post hoc / exploratory**. Nominal p-values are labelled as such.

## Scripts
| Script | Poster result |
|---|---|
| `run_salience_control.py` | 2: Salience <-> Control dFC variance vs CAPS improvement |
| `run_ramygdala_control.py` | 3: right amygdala <-> Control dFC variance vs CAPS improvement |
| `run_amygdala_vmpfc_change.py` | 4: change in left amygdala <-> vmPFC static FC vs CAPS improvement (not dFC) |
| `run_sn_dmn_ketamine.py` | 5: Salience <-> DMN dFC variance, per arm |
| `dfc_common.py` | shared loading, dFC, statistics |

## Data requirements and paths (what Amir needs)
Paths are never hard-coded in these scripts; they come from `../kpe_config.py` (repo root).
**kpe_config's default `KPE_BASE_DIR` is Zoie's personal path (a Hebrew-named folder on her Mac). It will not exist elsewhere: always set the environment variables below.**

1. `KPE_BASE_DIR` = a folder with this layout (kpe_config checks every item on import and stops with an error if one is missing):
```
<KPE_BASE_DIR>/
  timeseries_2026-08/
    tlv/                      (TAU_DIR)
      <any name>randomization<...>.xlsx     (file name must contain "randomization")
      anatomical/             (no GSR; 4 mm; used by results 2-5)
      global/                 (GSR; used by result 4 only)
        *_schaefer400_ts.csv, *_tian_s2_ts.csv, *_harvard_oxford_cortical_ts.csv, scrubbing_report_filtered.csv
    yale/                     (YALE_DIR; must exist)
      <any name>randomization<...>.xlsx     (required by kpe_config on import, even though these scripts use TLV only; the file is only located, not read here)
  organized_results/
    behavioural_yale_tlv.csv  (arms + CAPS; read by caps_source.py)
  output_data/                (created automatically when a script runs without --out)
```
   Yale imaging files are not needed, only the folder and a randomization xlsx in it.
2. `KPE_TIMESERIES_6MM` = the March 2026 TLV extraction (6 mm smoothing, anatomical), a folder with `sub-*_ses-MRI1_task-rest_run-1_{schaefer400,tian_s2}_ts.csv` and `scrubbing_report_filtered.csv`. Default `~/Downloads/timeseries/anatomical` (personal). Needed only for the 6 mm numbers (results 2, 3, 5); without it run with `--versions 4mm`.
3. `organized_results/behavioural_yale_tlv.csv` is **patient data (CAPS scores per subject). Share it privately with Amir. Never put it on GitHub.** Per-subject output files (`per_subject.csv`) are also patient-level: do not commit them.
4. Repo modules that must be in the commit: `kpe_config.py`, `caps_source.py`, `network_level_fc.py`, `atlas/Schaefer2018_400Parcels_7Networks_order.txt` (read at import by `network_level_fc`). **`network_level_fc.py` itself imports `amygdala_hippocampus_connectivity.py`, so that file is also required** (this package no longer imports it directly).
5. Python packages: see `requirements.txt` (tested with Python 3.14.8). `openpyxl` is not needed (kpe_config only locates the xlsx).

Example:
```
export KPE_BASE_DIR=/path/to/kpe
export KPE_TIMESERIES_6MM=/path/to/timeseries/anatomical
```

## How to run
From the repo root:
```
venv/bin/python dfc_poster/run_salience_control.py
venv/bin/python dfc_poster/run_ramygdala_control.py
venv/bin/python dfc_poster/run_amygdala_vmpfc_change.py
venv/bin/python dfc_poster/run_sn_dmn_ketamine.py
```
Options: `--out FOLDER` (default: timestamped folder under `kpe_config.OUTPUT_DATA_DIR`), `--no-check` (checks are on by default), `--versions 4mm` (skip the 6 mm data; 6 mm numbers and checks are then skipped; result 4 always uses 4 mm GSR/no-GSR).
Each script writes `per_subject.csv`, `results.csv`, `METHODS.txt`, prints a short summary, and with `--check` asserts the poster numbers (rounded) and exits non-zero on a mismatch.

## Expected numbers (checked by `--check`)
| Result | Value |
|---|---|
| 2 Salience-Control, 4 mm, pooled, n=30 | centred Spearman rho=-.66, permutation p~.0002 (this seed, 200,000 shuffles: .00019); per arm rho -.63 / -.54 / -.88 (ket0.5 / ket0.2 / midazolam); 6 mm pooled n=31: rho=-.48, perm p=.009 |
| 3 right amygdala-Control, 4 mm, pooled, n=30 | rho=-.56, permutation p~.002 (this seed, 200,000 shuffles: .0024); 6 mm rho=-.55 |
| 4 left amygdala-vmPFC change (day 30 - day 0) vs improvement, n=28 | GSR rho=+.52, p=.0045; no-GSR rho=+.41, p=.032 |
| 5 Salience-DMN, 6 mm, Pearson | ket0.5 n=10 r=-.73 p=.017; ket0.2 n=10 r=-.54 p=.111; midazolam n=11 r=-.25 p=.452; interaction ket0.5 vs midazolam p=.044, vs ket0.2 p=.035. 4 mm for transparency: ket0.5 r=-.58 p=.079; **4 mm midazolam Spearman rho=-.70 p=.024 (strongest arm at 4 mm)** |

With the fixed seed the output is deterministic; `--check` ranges guard against RNG changes between numpy versions (result 2 p in [.0001, .0003], result 3 p in [.0018, .0030]). Permutation p-values are Monte Carlo (200,000 shuffles, seed 2026; the n_perm actually used is written to results.csv and METHODS.txt), so the check accepts a range around the poster value.

## Methods summary
- **Subjects**: TLV, ses-1 (baseline), with an arm and CAPS. Excluded: sub-540, 576, 807, 949 (excluded per protocol); sub-900 (randomized but never scanned, no TLV imaging, so the exclusion changes nothing). The scripts also print subjects that have an arm and CAPS but no ses-1 scan (e.g. sub-342 at 4 mm). Scans with `scrubbed_volumes > 115` dropped; at least 120 retained volumes required for dFC. Duplicate rest runs: rest2 wins. Expected n: 4 mm 10/10/10, 6 mm 10/10/11 (ket0.5 / ket0.2 / midazolam; sub-024 dropped for motion).
- **Signals**: Schaefer-400 7-network means (SalVentAttn = Salience, Default = DMN, Cont = Control) through `schaefer_relabel()` (the 6 mm header is already official and passes through); Tian S2 amygdala = lAMY + mAMY per hemisphere.
- **dFC variance**: 60-TR sliding window (TR = 1 s), step 1, Pearson r per window, variance of the r values (ddof=1). Scrubbed volumes are not censored.
- **Outcome**: CAPS improvement = baseline - day 30 (positive = better), via `caps_source.load_dcaps` (sign flipped).
- **Pooled test (results 2, 3)**: variance and improvement are centred within each arm (arm mean subtracted), Spearman rho on the centred values; p from 200,000 within-arm permutations of the improvement scores (seeded, two-sided, (hits+1)/(N+1)). Centring stops differences between arm means from creating a correlation.
- **Interaction (result 5)**: OLS `improvement ~ var * arm` on two arms; the var:arm term tests whether the slopes differ.
- **Result 4 (static FC)**: left amygdala (lAMY-lh, mAMY-lh) with Harvard-Oxford "Frontal Medial Cortex"; Pearson per parcel pair, Fisher z, mean; change = ses-3 - ses-1. Scan rule here is `kpe_config.scan_drop()` (scrubbed > 115 or retained < 300 dropped), as in the original run `tlv_amyvmpfc_interaction_20261026_194620`.
- **FDR**: the Salience-Control q=.014 on the poster is a BH correction over the 67 dFC tests run on 2026-10-28. **It is not recomputed here**; these scripts report nominal p only.

## Caveats
- Everything is post hoc; small groups (n=9-11 per arm); nominal p-values.
- **Result 5 is fragile and not ketamine-specific.** It uses the 6 mm March extraction and Pearson only.
  - Spearman for ket0.5: p=.20.
  - At 4 mm the effect is weaker (ket0.5 r=-.58, p=.079) and **midazolam is the strongest arm (Spearman rho=-.70, p=.024; Pearson r=-.62, p=.054)**, the opposite of a ketamine-specific pattern. The interaction is not significant at 4 mm (p=.52 vs midazolam, p=.20 vs ket0.2).
  - The effect is absent with GSR.
  - Yale goes the opposite way (r=+.15, p=.67).
  - Leave-one-out: dropping sub-096 alone gives p=.095.
  - The script name `run_sn_dmn_ketamine.py` only says which arm is shown first; it does not imply ketamine specificity. Interaction output columns are `coef` and `p_interaction`.
- **Result 4: the change score largely reflects baseline connectivity.** In GSR, baseline FC vs improvement is rho=-.53 (p=.0035) while day-30 FC vs improvement is rho=+.01 (p=.95). The effect is also not seen across all arms: per arm, ket0.5 rho=+.66, ket0.2 +.52, midazolam -.38 (opposite sign), and centred within arm rho=+.25 (p=.21). No-GSR is the primary pipeline for the project (rho=+.41); GSR is shown on the poster. The script prints all these diagnostics.
- **Result 3 per arm**: ket0.2 is close to zero (rho=-.19). No right-vs-left amygdala test was run, so right-sided specificity is not shown.
- Result 4 output column `delta_fc` = static FC(ses-3) - FC(ses-1) (not dynamic FC).
- Result 2 per-arm Spearman: ket0.5 p=.05, ket0.2 p=.11, midazolam p=.0009 (nominal); the pooled number is driven partly by midazolam. 6 mm pooled for result 2 is weaker (rho=-.48).
