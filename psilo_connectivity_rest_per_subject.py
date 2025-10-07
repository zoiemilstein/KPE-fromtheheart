import os
import re
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_timeseries_table(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        if df[first_col].notna().sum() >= 3:
            unique_ratio = df[first_col].nunique(dropna=True) / df[first_col].notna().sum()
            if unique_ratio < 0.05:
                df = df.drop(columns=[first_col])
    return df

def remove_cerebellum_rois(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(cerebel|vermis)", flags=re.IGNORECASE)
    keep_cols = [c for c in df.columns if not pattern.search(str(c))]
    return df[keep_cols]

def pearson_connectivity(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")

def difference_matrix(corr_psilo: pd.DataFrame, corr_regular: pd.DataFrame) -> pd.DataFrame:
    common = sorted(set(corr_psilo.columns).intersection(set(corr_regular.columns)))
    corr_psilo = corr_psilo.loc[common, common]
    corr_regular = corr_regular.loc[common, common]
    return corr_psilo - corr_regular

def top_k_changes(diff_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    rois = diff_df.columns.tolist()
    pairs = list(itertools.combinations(range(len(rois)), 2))
    data = []
    for i, j in pairs:
        d = diff_df.iat[i, j]
        if pd.notna(d):
            data.append((rois[i], rois[j], float(d), abs(float(d))))
    data.sort(key=lambda x: x[3], reverse=True)
    top = data[: min(k, len(data))]
    return pd.DataFrame(top, columns=["roi_1", "roi_2", "delta_r", "abs_delta_r"])

def plot_top_changes(changes_df: pd.DataFrame, output_path: str, title: str) -> None:
    if changes_df.empty:
        return
    labels = [f"{r1}-{r2}" for r1, r2 in zip(changes_df["roi_1"], changes_df["roi_2"])]
    values = changes_df["delta_r"].to_numpy()
    order = np.argsort(np.abs(values))
    labels = [labels[i] for i in order]
    values = values[order]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), labels, fontsize=8)
    plt.xlabel("Δr (psilocybin − regular)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_between_session_connectivity(base_dir: str,
                                     file_regular: str,
                                     file_psilocybin: str,
                                     k_top: int = 20) -> dict:
    ts_regular = remove_cerebellum_rois(load_timeseries_table(file_regular))
    ts_psilo   = remove_cerebellum_rois(load_timeseries_table(file_psilocybin))
    corr_regular = pearson_connectivity(ts_regular)
    corr_psilo   = pearson_connectivity(ts_psilo)
    diff_df = difference_matrix(corr_psilo, corr_regular)
    top_changes = top_k_changes(diff_df, k=k_top)

    diff_csv = os.path.join(base_dir, "between_sessions_diff_matrix_psilo_minus_regular_noCerebellum.csv")
    top_csv  = os.path.join(base_dir, f"between_sessions_top{k_top}_delta_r_noCerebellum.csv")
    chart_png = os.path.join(base_dir, f"between_sessions_top{k_top}_delta_r_noCerebellum.png")

    diff_df.to_csv(diff_csv, index=True, encoding="utf-8")
    top_changes.to_csv(top_csv, index=False, encoding="utf-8")
    plot_top_changes(top_changes, chart_png, f"Top {k_top} connectivity changes (psilocybin − regular)")

    return {"diff_matrix_csv": diff_csv, "top_changes_csv": top_csv, "top_changes_chart_png": chart_png}

if __name__ == "__main__":
    base_dir = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\פסילו\data\sub001"
    file_regular = os.path.join(base_dir, "sub-001_ses-1_task-rest_aal_ts.csv")
    file_psilocybin = os.path.join(base_dir, "sub-001_ses-2_task-rest_aal_ts.csv")
    run_between_session_connectivity(base_dir, file_regular, file_psilocybin, k_top=20)
