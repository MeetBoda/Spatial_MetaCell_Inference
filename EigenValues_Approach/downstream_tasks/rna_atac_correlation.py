import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

font = {"size": 14}
plt.rc("font", **font)


def main(args):
    rna_metacell_adata = sc.read_h5ad(args.rna_metacell_path)

    atac_signac_adata = sc.read_h5ad(args.atac_signac_path)
    sc.pp.normalize_total(atac_signac_adata, target_sum=1e4)
    sc.pp.log1p(atac_signac_adata)

    metacell_id_adata = sc.read_h5ad(args.metacell_id_path)
    meta_ids = metacell_id_adata.obs["metacell"].values

    data = pd.DataFrame(
        columns=["Gene expression (logged)", "Chomatin accessibility (logged)"]
    )
    for i in np.unique(meta_ids):
        idx = np.where(meta_ids == i)[0]
        rna_mean = rna_metacell_adata[i].X.tolist()[0]
        atac_mean = np.mean(atac_signac_adata.X[idx], axis=0).tolist()[0]
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "Gene expression (logged)": rna_mean,
                        "Chomatin accessibility (logged)": atac_mean,
                    }
                ),
            ]
        )
    correlation = data.corr(method="spearman").iloc[0, 1]

    plt.figure(figsize=(7, 7), dpi=300)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font_scale=1.2)
    sns.scatterplot(
        x="Chomatin accessibility (logged)",
        y="Gene expression (logged)",
        data=data,
        s=1,
        alpha=0.1,
        linewidth=0,
    )

    plt.title("Spearman Correlation: %.3f" % correlation)

    plt.savefig(
        "./figures/RNA_ATAC_correlation.png", bbox_inches="tight", transparent=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--metacell_id_path", type=str)
    parser.add_argument("--rna_metacell_path", type=str)
    parser.add_argument("--atac_signac_path", type=str)
    parser.add_argument("--type_key", type=str, default="celltype")

    args = parser.parse_args()

    main(args)
