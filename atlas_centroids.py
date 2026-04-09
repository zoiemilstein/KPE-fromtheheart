#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from nilearn import datasets


# =========================================================
# SETTINGS
# =========================================================

OUT_DIR = Path("/Users/zoiemilstein/רפואה/מעבדה/kpe/atlas_centroids")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CORE FUNCTION
# =========================================================

def compute_centroids(atlas_img_path, labels, out_csv):
    print(f"\nProcessing atlas: {atlas_img_path}")

    img = nib.load(str(atlas_img_path))
    data = np.asarray(img.get_fdata())
    affine = img.affine

    # get unique parcel values (ignore background 0)
    values = sorted(int(v) for v in np.unique(data) if v != 0)

    rows = []
    for node_idx, atlas_val in enumerate(values):
        mask = data == atlas_val

        if not np.any(mask):
            continue

        # center of mass in voxel space
        voxel_center = center_of_mass(mask)

        # convert to MNI/world coordinates
        xyz = nib.affines.apply_affine(affine, voxel_center)

        row = {
            "node": node_idx,
            "atlas_value": atlas_val,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        }

        if labels is not None and node_idx < len(labels):
            row["label"] = str(labels[node_idx])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Saved {len(df)} centroids → {out_csv}")
    return df


# =========================================================
# SCHAEFER 400
# =========================================================

def build_schaefer():
    print("\nDownloading Schaefer 400 atlas...")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=400,
        yeo_networks=7,
        resolution_mm=2
    )

    atlas_path = atlas.maps
    labels = atlas.labels

    out_csv = OUT_DIR / "schaefer400_centroids.csv"

    df = compute_centroids(atlas_path, labels, out_csv)

    print("\nSanity check:")
    print(df.head())
    print(f"Total nodes: {len(df)}")

    return out_csv


# =========================================================
# TIAN S2
# =========================================================

def build_tian():
    print("\nDownloading Tian S2 atlas...")

    atlas = datasets.fetch_atlas_tian(
        subcortex="S2",
        resolution=2
    )

    atlas_path = atlas.maps

    # Tian labels sometimes come as dict or list depending on version
    labels = None
    if hasattr(atlas, "labels"):
        labels = atlas.labels

    out_csv = OUT_DIR / "tian_s2_centroids.csv"

    df = compute_centroids(atlas_path, labels, out_csv)

    print("\nSanity check:")
    print(df.head())
    print(f"Total nodes: {len(df)}")

    return out_csv


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n=== BUILDING CENTROIDS FILES ===")

    schaefer_file = build_schaefer()
    tian_file = build_tian()

    print("\nDONE")
    print("Files created:")
    print(f"- {schaefer_file}")
    print(f"- {tian_file}")


if __name__ == "__main__":
    main()