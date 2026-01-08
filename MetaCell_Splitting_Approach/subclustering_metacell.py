'''Here we are using Kmeans Clustering'''
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

# Paths
adata_path = "./scRNA_20kcell/sc_data_with_covet_eigdecomp.h5ad"   # raw scRNA annotated dataset
latent_path = "./sc_data_only_envi_latent.h5ad"   # (20000 × 512)
metacell_ids_path = "./scRNA_20kcell/spatialmetacell/BASELINE_EIGENDECOMP_500metacell_ids.h5ad"

# Outputs
output_spatial_label_path = "./scRNA_20kcell/spatialmetacell/spatial_metacell_labels_new_kmeans.npy"
output_adata_with_spatial_path = "./scRNA_20kcell/spatialmetacell/sc_adata_with_spatial_metacell_labels_new_kmeans.h5ad"

print("🔹 Loading data...")
adata = sc.read_h5ad(adata_path)
latent_adata = sc.read_h5ad(latent_path)
meta_ids = sc.read_h5ad(metacell_ids_path)

# Validate shapes
assert latent_adata.X.shape[0] == adata.X.shape[0], "❌ Latent embedding rows != adata cells"

# Insert latent embedding
adata.obsm["env_latent"] = latent_adata.X.copy()

# Insert metacell ids
adata.obs["metacell_id"] = meta_ids.obs["metacell"].astype(int).values
print("✔ Latent + metacell IDs added")

print("🔹 Performing KMeans clustering inside each MetaCell...")

spatial_labels = np.full(adata.n_obs, -1)
next_id = 0

for meta_id in tqdm(np.unique(adata.obs["metacell_id"])):

    cell_idx = np.where(adata.obs["metacell_id"] == meta_id)[0]
    latent_subset = adata.obsm["env_latent"][cell_idx]

    # Adaptive cluster count (same logic as your Leiden version)
    if len(cell_idx) < 10:
        cluster_count = 1
    elif len(cell_idx) < 30:
        cluster_count = 2
    else:
        cluster_count = min(5, len(cell_idx) // 10)

    if cluster_count == 1:
        # no clustering, assign one label
        spatial_labels[cell_idx] = next_id
        next_id += 1
        continue

    # ---- KMEANS CLUSTERING ----
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    local_labels = kmeans.fit_predict(latent_subset)

    # remap local 0..k-1 clusters → global spatial_metacell ID
    unique_local = np.unique(local_labels)
    mapping = {u: next_id + i for i, u in enumerate(unique_local)}

    for idx, lab in zip(cell_idx, local_labels):
        spatial_labels[idx] = mapping[lab]

    next_id += len(unique_local)

adata.obs["spatial_metacell"] = spatial_labels

print(f"✔ Total spatial metacells created: {np.unique(spatial_labels).shape[0]}")
np.save(output_spatial_label_path, spatial_labels)

adata.write(output_adata_with_spatial_path)
print("✔ Saved:", output_adata_with_spatial_path)



'''Here we are using Leiden Clustering'''
# import scanpy as sc
# import anndata as ad
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from tqdm import tqdm

# # Paths
# adata_path = "./scRNA_20kcell/sc_data_with_covet_eigdecomp.h5ad"   # raw scRNA annotated dataset
# latent_path = "./sc_data_only_envi_latent.h5ad"   # (20000 × 512)
# metacell_ids_path = "./scRNA_20kcell/spatialmetacell/BASELINE_EIGENDECOMP_500metacell_ids.h5ad"

# # Outputs
# output_spatial_label_path = "./scRNA_20kcell/spatialmetacell/spatial_metacell_labels_new.npy"
# output_adata_with_spatial_path = "./scRNA_20kcell/spatialmetacell/sc_adata_with_spatial_metacell_labels_new.h5ad"

# print("🔹 Loading data...")
# adata = sc.read_h5ad(adata_path)                 # raw scRNA
# latent_adata = sc.read_h5ad(latent_path)         # only env latent matrix
# meta_ids = sc.read_h5ad(metacell_ids_path)       # metacell assignments

# # Validate shapes
# assert latent_adata.X.shape[0] == adata.X.shape[0], "❌ Latent embedding rows != adata cells"

# # Insert latent embedding
# adata.obsm["env_latent"] = latent_adata.X.copy()

# # Insert metacell ids
# adata.obs["metacell_id"] = meta_ids.obs["metacell"].astype(int).values
# print("✔ Latent + metacell IDs added")

# print("🔹 Performing Leiden clustering inside each MetaCell...")

# spatial_labels = np.full(adata.n_obs, -1)
# next_id = 0

# for meta_id in tqdm(np.unique(adata.obs["metacell_id"])):
#     cell_idx = np.where(adata.obs["metacell_id"] == meta_id)[0]
#     sub = adata[cell_idx].copy()
    
#     if len(cell_idx) < 10:
#         # too small → no clustering
#         spatial_labels[cell_idx] = next_id
#         next_id += 1
#         continue

#     # Use latent embedding
#     sub.obsm["X_latent"] = adata.obsm["env_latent"][cell_idx]

#     # Build graph + Leiden
#     sc.pp.neighbors(sub, use_rep="X_latent", n_neighbors=10)
    
#     # adaptive resolution
#     if len(cell_idx) < 50:
#         res = 0.3
#     elif len(cell_idx) < 150:
#         res = 0.5
#     else:
#         res = 1.0

#     sc.tl.leiden(sub, resolution=res, key_added="spatial_local")

#     # assign clusters
#     local_labels = sub.obs["spatial_local"].astype(int).values
    
#     unique_local = np.unique(local_labels)
#     mapping = {u: next_id + i for i, u in enumerate(unique_local)}
    
#     for idx, lab in zip(cell_idx, local_labels):
#         spatial_labels[idx] = mapping[lab]

#     next_id += len(unique_local)

# adata.obs["spatial_metacell"] = spatial_labels

# print(f"✔ Total spatial metacells created: {np.unique(spatial_labels).shape[0]}")
# np.save(output_spatial_label_path, spatial_labels)

# # Save intermediate annotated adata
# adata.write(output_adata_with_spatial_path)
# print("✔ Saved:", output_adata_with_spatial_path)

