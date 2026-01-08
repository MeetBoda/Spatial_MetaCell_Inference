# Code to append eigenvalues in scRNA data
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from anndata import AnnData


# Load the AnnData file
adata = sc.read_h5ad("sc_data_with_covet_eigdecomp.h5ad")


# Extract eigenvalue matrix
eigvals = adata.obsm["covet_eigvals"]  # shape: (20000, 25)


# Combine matrices
if issparse(adata.X):
    covet_sparse = csr_matrix(eigvals)
    X_combined = hstack([adata.X, covet_sparse])
else:
    X_combined = np.hstack([adata.X, eigvals])


# Combine variable names
original_var = adata.var.copy()
covet_var = pd.DataFrame(index=[f"covet_feat_{i}" for i in range(eigvals.shape[1])])
combined_var = pd.concat([original_var, covet_var])


# Create new AnnData
adata_new = AnnData(X=X_combined, obs=adata.obs.copy(), var=combined_var)
# Save to new file
adata_new.write_h5ad("sc_data_with_eigenvalues_as_gene.h5ad")
