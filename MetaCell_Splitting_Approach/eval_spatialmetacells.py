import scanpy as sc
from eval_utils_spatial_metacells import (
    pairwise_correlation,
    compactness,
    separation,
    plot_compactness_separation,
    plot_metacell_umap,
    plot_metacell_size,
)

# === Paths ===
spatial_metacell_path = "./scRNA_20kcell/spatialmetacell/SpatialMetaCell_RNA_LATENT_Combined_new_kmeans.h5ad"
raw_count_path = "./scRNA_20kcell/sc_data_with_covet_eigdecomp.h5ad"
save_name = "SpatialMetaCell_Analysis_USING_LATENT_of_scRNA_Kmeans"

# === Load Data ===
adata = sc.read_h5ad(spatial_metacell_path)
raw_adata = sc.read_h5ad(raw_count_path)

# === Assign 'metacell' label for convenience
adata.obs['metacell'] = adata.obs['metacell_id']

# === Plot metrics
plot_compactness_separation(raw_adata, adata, save_name)

# === Compute neighbors and UMAP using latent embedding
if 'envi_latent' in adata.obsm:
    adata.obsm['X_pca'] = adata.obsm['envi_latent']
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
else:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# === Plot visualizations
plot_metacell_umap(adata, save_name)
plot_metacell_size(adata, save_name)

