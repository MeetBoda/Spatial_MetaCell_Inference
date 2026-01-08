import warnings

warnings.simplefilter("ignore")
import torch
import argparse
import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from harmonypy import compute_lisi


def main(args):
    original_adata = sc.read_h5ad(args.original_path)
    batch_ids = original_adata.obs[args.batch_key].cat.codes
    batch_map = {
        i: original_adata.obs[args.batch_key].cat.categories[i]
        for i in range(batch_ids.max() + 1)
    }
    batch_ids = torch.from_numpy(batch_ids.values).long()
    batch_one_hot = torch.zeros(batch_ids.shape[0], batch_ids.max() + 1)
    batch_one_hot.scatter_(1, batch_ids.unsqueeze(1), 1)

    metacell_id_adata = sc.read_h5ad(args.metacell_id_path)
    meta_ids = metacell_id_adata.obs["metacell"].values
    non_empty_metacell = np.zeros(meta_ids.max() + 1).astype(bool)
    non_empty_metacell[np.unique(meta_ids)] = True
    batch_meta = (
        torch.stack(
            [
                batch_one_hot[meta_ids == i].mean(dim=0)
                for i in range(meta_ids.max() + 1)
            ]
        )
        .argmax(dim=1)
        .numpy()
    )
    batch_meta = np.array([batch_map[i] for i in batch_meta])
    batch_meta = batch_meta[non_empty_metacell]

    metacell_adata = sc.read_h5ad(args.metacell_path)
    metacell_adata.obs[args.batch_key] = batch_meta
    sc.pp.pca(metacell_adata)

    print("=======Performing Harmony Integration=======")
    harmony_out = harmonypy.run_harmony(
        metacell_adata.obsm["X_pca"],
        metacell_adata.obs,
        args.batch_key,
        max_iter_harmony=30,
        nclust=15,
        theta=10.0,
        verbose=False,
    )
    metacell_adata.obsm["X_pca_harmony"] = harmony_out.Z_corr.T

    lisi_score = compute_lisi(
        metacell_adata.obsm["X_pca_harmony"],
        pd.DataFrame(metacell_adata.obs),
        label_colnames=[args.type_key, args.batch_key],
        perplexity=10,
    )
    cell_types = np.unique(metacell_adata.obs[args.type_key].values)
    avg_lisi_score = np.zeros((len(cell_types), 2))
    for i, cell_type in enumerate(cell_types):
        indices = metacell_adata.obs[args.type_key] == cell_type
        avg_lisi_score[i] = lisi_score[indices].mean(axis=0)
    clisi = avg_lisi_score[:, 0]
    ilisi = avg_lisi_score[:, 1]
    data = pd.DataFrame(columns=["Metric", "Score"])
    for score in clisi:
        score = 1 - (score - 1) / len(cell_types)
        data = pd.concat(
            [data, pd.DataFrame({"Metric": ["1 - cLISI"], "Score": [score]})]
        )
    for score in ilisi:
        score = (score - 1) / len(batch_map)
        data = pd.concat([data, pd.DataFrame({"Metric": ["iLISI"], "Score": [score]})])
    plt.figure(figsize=(4, 7), dpi=300)
    sns.set_theme(style="ticks", font_scale=1.0)

    sns.boxplot(
        data=data,
        y="Score",
        x="Metric",
        saturation=0.55,
        fliersize=0.5,
        linewidth=0.5,
        width=0.87,
    )

    plt.title("Metacell Metric")
    plt.tight_layout()

    plt.savefig("./figures/lisi_metric.png", transparent=False)
    plt.close()

    sc.set_figure_params(figsize=(7, 7), dpi=300)
    sc.pp.neighbors(
        metacell_adata, use_rep="X_pca_harmony", metric="cosine", n_neighbors=7
    )
    sc.tl.umap(metacell_adata)
    sc.pl.umap(
        metacell_adata,
        color=[args.type_key],
        save="_harmony_type.png",
        palette=sns.color_palette(
            "husl", np.unique(metacell_adata.obs[args.type_key].values).size
        ),
        show=False,
    )
    sc.pl.umap(
        metacell_adata,
        color=[args.batch_key],
        save="_harmony_batch.png",
        palette=sns.color_palette(
            "hls", np.unique(metacell_adata.obs[args.batch_key].values).size
        ),
        show=False,
    )

    metacell_adata.write_h5ad(args.metacell_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--original_path", type=str)
    parser.add_argument("--metacell_path", type=str)
    parser.add_argument("--metacell_id_path", type=str)
    parser.add_argument("--type_key", type=str, default="celltype")
    parser.add_argument("--batch_key", type=str, default="batch")

    args = parser.parse_args()

    main(args)
