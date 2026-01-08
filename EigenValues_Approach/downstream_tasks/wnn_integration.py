import warnings

warnings.filterwarnings("ignore")

import argparse
import muon as mu
import scanpy as sc
import seaborn as sns
from muon import MuData


def main(args):
    rna_metacell_adata = sc.read_h5ad(args.rna_metacell_path)
    adt_metacell_adata = sc.read_h5ad(args.adt_metacell_path)

    mdata = MuData({"RNA": rna_metacell_adata, "ADT": adt_metacell_adata})
    sc.pp.pca(mdata["RNA"], n_comps=30)
    sc.pp.pca(mdata["ADT"], n_comps=18)
    sc.pp.neighbors(mdata["RNA"], n_neighbors=3)
    sc.pp.neighbors(mdata["ADT"], n_neighbors=3)

    # Calculate weighted nearest neighbors
    mu.pp.neighbors(mdata, key_added="wnn", n_multineighbors=7)
    mu.tl.umap(mdata, neighbors_key="wnn", random_state=42)
    sc.set_figure_params(figsize=(7, 7), dpi=300)
    mu.pl.umap(
        mdata,
        color=["RNA:" + args.type_key],
        palette=sns.color_palette("husl", 27),
        save="_wnn.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--rna_metacell_path", type=str)
    parser.add_argument("--adt_metacell_path", type=str)
    parser.add_argument("--type_key", type=str, default="celltype")

    args = parser.parse_args()

    main(args)
