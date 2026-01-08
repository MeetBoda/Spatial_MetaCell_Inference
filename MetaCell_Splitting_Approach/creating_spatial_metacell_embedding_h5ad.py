import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

# Load the SpatialMetaCell × Features matrix
adata_mc = sc.read_h5ad("./scRNA_20kcell/spatialmetacell/SpatialMetaCell_RNA_LATENT_Combined_new_kmeans.h5ad")

print("Input:", adata_mc.shape)

# -------------------------------------------------
# STEP 1 — Normalize + PCA
# -------------------------------------------------
adata_embed = adata_mc.copy()
sc.pp.scale(adata_embed, max_value=10)
sc.tl.pca(adata_embed, n_comps=50)

# -------------------------------------------------
# STEP 2 — Build kNN Graph
# -------------------------------------------------
sc.pp.neighbors(
    adata_embed,
    use_rep="X_pca",
    n_neighbors=20,
)

# -------------------------------------------------
# STEP 3 — Diffusion Map embeddings
# -------------------------------------------------
sc.tl.diffmap(adata_embed)

# Extract full diffusion map matrix
diff = adata_embed.obsm["X_diffmap"]
n_total = diff.shape[1]           # total dims in X_diffmap
n_use = min(32, n_total - 1)      # usable dims (skip first trivial)

print(f"Total diffusion dims: {n_total}")
print(f"Using {n_use} dims")

# extract only available eigenvectors
X = diff[:, 1 : 1 + n_use]

print("Final embedding shape:", X.shape)

# -------------------------------------------------
# STEP 4 — Save to AnnData
# -------------------------------------------------
var_names = [f"embed_{i}" for i in range(n_use)]

adata_out = ad.AnnData(
    X=X,
    obs=adata_mc.obs.copy(),
    var=pd.DataFrame(index=var_names)
)

adata_out.obs_names = adata_mc.obs_names

# Save
output_path = "./scRNA_20kcell/spatialmetacell/SpatialMetaCell_Embedding32_kmeans.h5ad"
adata_out.write(output_path)

print("✔ Saved embedding:", output_path)
