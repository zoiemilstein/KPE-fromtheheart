# aal_roi_plotly_fsavg_3d_seed.py
# Interactive 3D fsaverage cortex colored by AAL ROI p-values (ROI-level).
# - Hover shows the EXACT ROI name from your CSV.
# - Optional seed filtering (e.g., only connections from Amygdala_L/R).
# - Title includes the chosen seed (original CSV label) and p-threshold.
# - Offline HTML (double-click) + optional PNG export.

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
import plotly.graph_objects as go

# ========================= USER INPUT =========================
CSV_PATH = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\KPE\new_data\t_test_ses_1_3\between_group_followup_minus_baseline_amygdala.csv"
FILTER_BY_SEED = "Amygdala_L"

P_THRESH = None  # show only ROIs with p < P_THRESH; set None to show all
COLORSCALE = "Amp"  # "Magma","Viridis","Turbo","Plasma","Inferno","Cividis", ...
OPACITY = 1.0
HEMISPHERE_SHIFT = 55  # mm separation between hemispheres


OUT_HTML = "fsavg_roi_pmap_3d.html"

PNG_PATH = "fsavg_roi_pmap_3d.png"
# =============================================================

EPS = 1e-16


def fetch_aal_atlas():
    """Fetch and prepare AAL atlas data"""
    print("Fetching AAL (SPM12)...")
    aal = datasets.fetch_atlas_aal(version="SPM12")  # MNI152 2mm
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)

    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    indices_raw = list(aal["indices"]) if "indices" in aal else sorted(int(v) for v in np.unique(atlas) if v > 0)[
                                                                :len(labels_raw)]
    # normalized AAL label -> voxel value in atlas (e.g., 2001..9170)
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}

    print(f"Atlas shape: {atlas.shape} ")
    return aal, atlas_img, atlas, name_to_index

def load_and_filter_csv():
    """Load CSV data and apply filtering"""
    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    roi_col = "region"  # target ROI column (e.g., "region" / "label_name" / "ROI")
    p_value_col = "p_value"  # p-value column
    seed_col = "seed"

    display_seed_label = None
    df = df[df[seed_col] == FILTER_BY_SEED]

    display_map = df.set_index(roi_col)[roi_col].to_dict()
    # threshold (optional)
    if P_THRESH is not None:
        before_count = len(df)
        df = df[df[p_value_col] < P_THRESH]
        after_count = len(df)
        print(f"P-threshold filtering: {before_count} → {after_count} ROIs (p < {P_THRESH})")
        if after_count == 0:
            raise RuntimeError(f"No ROIs passed the p-threshold of {P_THRESH}.")

    return df, roi_col, p_value_col, seed_col, display_seed_label, display_map

def create_roi_mappings(df, name_to_index, roi_col, p_value_col, display_map):
    """Create ROI mappings from CSV data"""
    print("Mapping CSV p-values to AAL indices")
    roi_index_to_p_value = {}
    roi_index_to_intensity = {}
    roi_index_to_display = {}

    for _, r in df.iterrows():
        roi = r[roi_col]
        p_value = float(r[p_value_col])
        index = name_to_index.get(roi)
        roi_index_to_p_value[index] = p_value
        roi_index_to_intensity[index] = max(-np.log10(p_value + EPS), 0.0)
        roi_index_to_display[index] = display_map.get(roi)  # exact CSV ROI for hover

    return roi_index_to_p_value, roi_index_to_intensity, roi_index_to_display

def fetch_fsaverage_surfaces():
    """Fetch fsaverage surfaces"""
    print("Fetching fsaverage surfaces (pial + white)")
    fsavg = datasets.fetch_surf_fsaverage()
    pial_L, pial_R = fsavg["pial_left"], fsavg["pial_right"]
    white_L, white_R = fsavg["white_left"], fsavg["white_right"]
    return pial_L, pial_R, white_L, white_R

def sample_labels(atlas_img, white_mesh, pial_mesh):
    """Sample atlas labels onto surface vertices"""
    try:
        lbl = surface.vol_to_surf(
            atlas_img, pial_mesh,
            inner_mesh=white_mesh, kind="line", n_samples=25, interpolation="nearest"
        )
    except TypeError:
        lbl = surface.vol_to_surf(atlas_img, pial_mesh)
    lbl = np.rint(np.asarray(lbl)).astype(np.int32)
    lbl[lbl < 0] = 0
    return lbl

def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_intensity, roi_index_to_p_value, roi_index_to_display):
    """Build vertex data for surface visualization"""
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    pvals = np.full(coords.shape[0], np.nan, dtype=float)
    names = np.empty(coords.shape[0], dtype=object)

    for val, inten in roi_index_to_intensity.items():
        mask = (labels_on_vertices == val)
        if not np.any(mask):
            continue
        intens[mask] = inten
        pvals[mask] = roi_index_to_p_value[val]
        names[mask] = roi_index_to_display.get(val, "")

    return coords, faces, intens, pvals, names

def shift_hemispheres(coords_L, coords_R):
    """Shift hemispheres for clean separation"""
    coords_L_shift = coords_L.copy()
    coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy()
    coords_R_shift[:, 0] += HEMISPHERE_SHIFT
    return coords_L_shift, coords_R_shift

def calculate_color_scale(intens_L, intens_R):
    """Calculate color scale range from intensities"""
    all_intens = np.concatenate([
        intens_L[~np.isnan(intens_L)], intens_R[~np.isnan(intens_R)]
    ])
    if all_intens.size == 0 or np.nanmax(all_intens) == 0:
        raise RuntimeError("All intensities are zero/NaN after filtering; relax P_THRESH or adjust inputs.")
    cmin, cmax = float(np.nanmin(all_intens)), float(np.nanmax(all_intens))
    return cmin, cmax

def create_background_layers(fig, coords_L_shift, faces_L, coords_R_shift, faces_R):
    """Create background gray surfaces"""
    lighting = dict(ambient=0.35, diffuse=0.6, specular=0.2, roughness=0.8, fresnel=0.2)
    
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        color="lightgray", opacity=0.25, name="background L",
        hoverinfo="skip", showscale=False, lighting=dict(ambient=0.5, diffuse=0.5)
    ))
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        color="lightgray", opacity=0.25, name="background R",
        hoverinfo="skip", showscale=False, lighting=dict(ambient=0.5, diffuse=0.5)
    ))

def create_hover_data(names_L, p_L, names_R, p_R):
    """Create hover data arrays"""
    custom_L = np.stack([
        names_L.astype(str),
        p_L
    ], axis=1)
    custom_R = np.stack([
        names_R.astype(str),
        p_R
    ], axis=1)
    hover_tmpl = "<b>%{customdata[0]}</b><br>p = %{customdata[1]:.3g}<extra></extra>"
    return custom_L, custom_R, hover_tmpl

def create_roi_layers(fig, coords_L_shift, faces_L, intens_L, coords_R_shift, faces_R, intens_R, 
                      cmin, cmax, custom_L, custom_R, hover_tmpl):
    """Create colored ROI layers"""
    lighting = dict(ambient=0.35, diffuse=0.6, specular=0.2, roughness=0.8, fresnel=0.2)
    
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=cmin, cmax=cmax, colorscale=COLORSCALE,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=True, name="Left hemisphere",
        hovertemplate=hover_tmpl, customdata=custom_L,
        colorbar=dict(
            title="-log10(p)", len=0.80,
            tickvals=[-np.log10(x) for x in (0.05, 0.01, 0.001)],
            ticktext=["0.05", "0.01", "0.001"]
        )
    ))
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=cmin, cmax=cmax, colorscale=COLORSCALE,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=False, name="Right hemisphere",
        hovertemplate=hover_tmpl, customdata=custom_R
    ))

def create_title(seed_col, display_seed_label):
    """Create dynamic title"""
    title_parts = ["fsaverage cortex – AAL ROI p-map"]
    if seed_col is not None and FILTER_BY_SEED is not None:
        title_parts.append(f"seed: {display_seed_label or FILTER_BY_SEED}")
    if P_THRESH is not None:
        title_parts.append(f"p < {P_THRESH:g}")
    fig_title = " | ".join(title_parts)
    return fig_title

def configure_layout(fig, fig_title):
    """Configure figure layout"""
    fig.update_layout(
        title=fig_title,
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02)
    )

def save_outputs(fig):
    """Save HTML and PNG outputs"""
    # offline, self-contained HTML
    fig.write_html(OUT_HTML, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Wrote: {OUT_HTML}")

    try:
        fig.write_image(PNG_PATH, width=2000, height=1400, scale=2)
        print(f"Wrote: {PNG_PATH}")
    except Exception as e:
        print("PNG export needs 'kaleido' (pip install -U kaleido). Error:", e)

def main():
    # Load atlas
    aal, atlas_img, atlas, name_to_index = fetch_aal_atlas()
    
    # Load and filter data
    df, roi_col, p_value_col, seed_col, display_seed_label, display_map = load_and_filter_csv()
    
    # Create ROI mappings
    roi_index_to_p_value, roi_index_to_intensity, roi_index_to_display = create_roi_mappings(
        df, name_to_index, roi_col, p_value_col, display_map
    )
    
    # Fetch surfaces
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()
    
    # Sample labels
    print("Sampling labels onto surface vertices...")
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)
    
    # Build vertex data
    coords_L, faces_L, intens_L, p_L, names_L = build_vertex_data(
        pial_L, lbl_L, roi_index_to_intensity, roi_index_to_p_value, roi_index_to_display
    )
    coords_R, faces_R, intens_R, p_R, names_R = build_vertex_data(
        pial_R, lbl_R, roi_index_to_intensity, roi_index_to_p_value, roi_index_to_display
    )
    
    # Shift hemispheres
    coords_L_shift, coords_R_shift = shift_hemispheres(coords_L, coords_R)
    
    # Calculate color scale
    cmin, cmax = calculate_color_scale(intens_L, intens_R)
    
    # Create figure
    fig = go.Figure()
    
    # Create background layers
    create_background_layers(fig, coords_L_shift, faces_L, coords_R_shift, faces_R)
    
    # Create hover data
    custom_L, custom_R, hover_tmpl = create_hover_data(names_L, p_L, names_R, p_R)
    
    # Create ROI layers
    create_roi_layers(fig, coords_L_shift, faces_L, intens_L, coords_R_shift, faces_R, intens_R, 
                      cmin, cmax, custom_L, custom_R, hover_tmpl)
    
    # Create title and configure layout
    fig_title = create_title(seed_col, display_seed_label)
    configure_layout(fig, fig_title)
    
    # Save outputs
    save_outputs(fig)


if __name__ == "__main__":
    main()
