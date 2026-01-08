import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from alive_progress import alive_bar
from sklearn.metrics import balanced_accuracy_score

def pearson_pytorch(A, B):
    if isinstance(A, sparse.csr_matrix) or isinstance(A, sparse.csc_matrix):
        A = A.toarray()
    if isinstance(B, sparse.csr_matrix) or isinstance(B, sparse.csc_matrix):
        B = B.toarray()
    A = torch.from_numpy(A).half()
    B = torch.from_numpy(B).half()
    if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()

    A_mean = A - A.mean(dim=1, keepdim=True)
    B_mean = B - B.mean(dim=1, keepdim=True)
    A_std = A_mean.norm(dim=1)
    B_std = B_mean.norm(dim=1)
    cov_matrix = torch.mm(A_mean, B_mean.t())
    correlation_matrix = cov_matrix / torch.outer(A_std, B_std)

    return correlation_matrix.cpu().numpy()

def pairwise_correlation(data, save_name):
    cell_num = data.shape[0]
    save_path = f"./save/{save_name}_{cell_num}cells_correlation.npy"
    try:
        corr = np.load(save_path)
        print("Correlation Loaded from:", save_path)
    except:
        print("Computing Pairwise Correlation...")
        corr = np.zeros((cell_num, cell_num), dtype=np.float16)
        chunk_size = 5000
        chunks_num = cell_num // chunk_size + 1 if cell_num % chunk_size != 0 else 0
        total_steps = chunks_num * chunks_num
        with alive_bar(total_steps, enrich_print=False) as bar:
            for i in range(0, cell_num, chunk_size):
                for j in range(0, cell_num, chunk_size):
                    corr[i:i+chunk_size, j:j+chunk_size] = pearson_pytorch(
                        data[i:i+chunk_size], data[j:j+chunk_size])
                    bar()
        np.save(save_path, corr)
        print("Correlation Saved to:", save_path)
    return corr

def compactness(corr, spatial_metacell_assign):
    spatial_metacell_ids = np.unique(spatial_metacell_assign)
    compactness_scores = []
    for i in spatial_metacell_ids:
        idx = np.where(spatial_metacell_assign == i)[0]
        if len(idx) == 0:
            continue
        compactness_scores.append(np.mean(corr[idx][:, idx]))
    return compactness_scores

def separation(corr, spatial_metacell_assign):
    spatial_metacell_ids = np.unique(spatial_metacell_assign)
    separation_scores = []
    for i in spatial_metacell_ids:
        idx = np.where(spatial_metacell_assign == i)[0]
        complementary_idx = np.where(spatial_metacell_assign != i)[0]
        if len(idx) == 0:
            continue
        separation_scores.append(np.mean(1 - corr[idx][:, complementary_idx].max(axis=1)))
    return separation_scores

def plot_compactness_separation(raw_adata, adata_spatial, save_name):
    adata_spatial.obs['metacell'] = adata_spatial.obs['metacell_id']
    raw_adata = sc.AnnData(raw_adata.X.copy())
    sc.pp.normalize_total(raw_adata, target_sum=1e4)
    sc.pp.log1p(raw_adata)

    corr = pairwise_correlation(raw_adata.X, save_name)
    spatial_assignments = adata_spatial.obs['metacell']

    compactness_scores = compactness(corr, spatial_assignments)
    separation_scores = separation(corr, spatial_assignments)

    print("* Avg Compactness:", np.mean(compactness_scores))
    print("* Avg Separation:", np.mean(separation_scores))

    metrics = pd.DataFrame({
        'Score': compactness_scores + separation_scores,
        'Metric': ['Compactness'] * len(compactness_scores) + ['Separation'] * len(separation_scores)
    })

    plt.figure(figsize=(4, 6))
    sns.boxplot(data=metrics, x='Metric', y='Score', saturation=0.6, width=0.7)
    plt.title('Spatial MetaCell Metrics')
    plt.tight_layout()
    plt.savefig(f"./figures/{save_name}_metric.png")
    plt.close()


def plot_metacell_umap(adata, save_name, meta_size=50, cell_size=0.5):
    adata.obs['metacell'] = adata.obs['metacell_id'].astype("category")

    # Prepare UMAP coordinates with metacell labels
    umap_df = (
        pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        .set_index(adata.obs_names)
        .join(adata.obs["metacell"])
        .reset_index(drop=True)
    )

    # Compute metacell centroids
    centroids = umap_df.groupby("metacell")[["UMAP1", "UMAP2"]].mean().reset_index()

    plt.figure(figsize=(8, 7), dpi=300)
    sns.set_theme(style="ticks")

    # Plot individual cells (colored by metacell)
    sns.scatterplot(
        data=umap_df, x="UMAP1", y="UMAP2", hue="metacell",
        s=cell_size, linewidth=0, alpha=0.6, legend=False
    )

    # Plot metacell centroids with same color hue
    sns.scatterplot(
        data=centroids, x="UMAP1", y="UMAP2", hue="metacell",
        s=meta_size, edgecolor="black", linewidth=0.8, legend=False
    )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of Spatial MetaCells")
    plt.tight_layout()
    plt.savefig(f"./figures/{save_name}_umap.png")
    plt.close()

def plot_metacell_size(adata, save_name):
    adata.obs['metacell'] = adata.obs['metacell_id']
    sizes = adata.obs['metacell'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.histplot(sizes, bins=range(1, sizes.max() + 2), kde=True, color='skyblue', edgecolor='black')
    plt.title("Distribution of Spatial MetaCell Sizes")
    plt.xlabel("Number of Cells per Spatial MetaCell")
    plt.ylabel("Count of Spatial MetaCells")
    plt.tight_layout()
    plt.savefig(f"./figures/{save_name}_size.png")
    plt.close()






