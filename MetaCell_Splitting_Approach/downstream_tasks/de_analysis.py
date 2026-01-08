import warnings

warnings.filterwarnings("ignore")

import argparse
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):
    metacell_adata = sc.read_h5ad(args.metacell_path)
    sc.tl.rank_genes_groups(
        metacell_adata, args.group_key, method="wilcoxon", key_added="wilcoxon"
    )
    names = metacell_adata.uns["wilcoxon"]["names"].tolist()
    dedf = sc.get.rank_genes_groups_df(metacell_adata, key="wilcoxon", group=None)

    selected_genes = []
    for i in range(len(names[0])):
        for j in range(args.top_genes):
            if names[j][i] not in selected_genes:
                selected_genes.append(names[j][i])

    dedf = dedf.pivot(index="group", columns="names", values="logfoldchanges")
    dedf = dedf[selected_genes]

    # DE values
    plt.figure(figsize=(15, 3), dpi=300)
    ax = sns.heatmap(
        dedf,
        vmin=-3,
        vmax=3,
        cmap=sns.color_palette("vlag", as_cmap=True),
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.subplots_adjust(left=0.25, right=1.0, top=0.95, bottom=0.25)
    plt.savefig("./figures/de_heatmap_" + args.group_key + "_value.png")
    plt.close()

    # DE ranks
    for i in range(dedf.shape[0]):
        dedf.iloc[i] = dedf.iloc[i].argsort().argsort()
    plt.figure(figsize=(15, 3), dpi=300)
    ax = sns.heatmap(
        dedf,
        vmin=1,
        vmax=len(selected_genes),
        cmap=sns.color_palette("Oranges", as_cmap=True),
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.subplots_adjust(left=0.25, right=1.0, top=0.95, bottom=0.25)
    plt.savefig("./figures/de_heatmap_" + args.group_key + "_rank.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--metacell_path", type=str)
    parser.add_argument("--group_key", type=str, default="celltype")
    parser.add_argument("--top_genes", type=int, default=10)

    args = parser.parse_args()

    main(args)
