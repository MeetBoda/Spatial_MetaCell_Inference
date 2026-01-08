"""
Generate Spatial MetaCell × (RNA + latent) feature matrix.

Input:
    1. BASELINE_EIGENDECOMP_RNA_500metacell.h5ad
        → MetaCell × gene features

    2. sc_adata_with_spatial_metacell_labels_new.h5ad
        → Each cell has: metacell_id, spatial_metacell, env_latent

Output:
    1. SpatialMetaCell_RNA_LATENT_Combined.h5ad
        → (#SpatialMetaCells × (#genes + #latent))

    Includes metadata:
        obs["metacell_id"]
        obs["n_cells"]
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

# ================================
# Paths
# ================================
meta_feature_path = "./scRNA_20kcell/spatialmetacell/BASELINE_EIGENDECOMP_RNA_500metacell.h5ad"
adata_spatial_path = "./scRNA_20kcell/spatialmetacell/sc_adata_with_spatial_metacell_labels_new_kmeans.h5ad"
output_path = "./scRNA_20kcell/spatialmetacell/SpatialMetaCell_RNA_LATENT_Combined_new_kmeans.h5ad"

print("🔹 Loading MetaCell RNA features…")
meta_features = sc.read_h5ad(meta_feature_path)

print("🔹 Loading spatial-metacell annotated data…")
adata = sc.read_h5ad(adata_spatial_path)

# ================================
# Step 1 — Map spatial_metacell → metacell_id
# ================================
print("🔹 Mapping SpatialMetaCell → MetaCell IDs…")

spatial_to_metacell = adata.obs.groupby("spatial_metacell")["metacell_id"].first()

# ================================
# Step 2 — Extract gene features for each spatial_metacell
# ================================
print("🔹 Extracting RNA features…")

if meta_features.obs.empty:
    # meta_features.obs_names are the metacell IDs
    rna_df = pd.DataFrame(meta_features.X)
    rna_df.index = meta_features.obs_names.astype(int)
else:
    # meta_features.obs['metacell'] column exists
    rna_df = pd.DataFrame(meta_features.X)
    rna_df.index = meta_features.obs["metacell"].astype(int)

# Subset by metacell IDs used by every spatial_metacell
try:
    rna_sub = rna_df.loc[spatial_to_metacell.values].copy()
except KeyError as e:
    raise KeyError(
        f"❌ ERROR: Some metacell IDs are missing in the RNA feature file.\n{e}"
    )

rna_sub.index = spatial_to_metacell.index  # rename index to spatial_metacell IDs

# ================================
# Step 3 — Average latent embedding per spatial_metacell
# ================================
print("🔹 Aggregating latent embeddings…")

latent_df = pd.DataFrame(adata.obsm["env_latent"], index=adata.obs_names)
latent_df["spatial_metacell"] = adata.obs["spatial_metacell"].values

latent_avg = latent_df.groupby("spatial_metacell").mean()

# ================================
# Step 4 — Combine gene features + latent features
# ================================
print("🔹 Combining RNA + latent features…")

combined = pd.concat([rna_sub, latent_avg], axis=1)

# ================================
# Step 5 — Build metadata
# ================================
print("🔹 Creating metadata for each SpatialMetaCell…")

obs = pd.DataFrame(index=combined.index)
obs["metacell_id"] = spatial_to_metacell.values
obs["n_cells"] = adata.obs["spatial_metacell"].value_counts().sort_index().values

# ================================
# Step 6 — Create final AnnData
# ================================
print("🔹 Creating AnnData object…")

# variable names
var_names = (
    [f"gene_{i}" for i in range(rna_sub.shape[1])] +
    [f"latent_{i}" for i in range(latent_avg.shape[1])]
)

adata_out = ad.AnnData(
    X=combined.values,
    obs=obs,
    var=pd.DataFrame(index=var_names)
)

adata_out.obs_names = combined.index.astype(str)


# ================================
# Step 7 — Save Output
# ================================
adata_out.write(output_path)

print("Saved SpatialMetaCell × (RNA + latent) matrix!")
print("File:", output_path)
print("Shape:", adata_out.shape)
