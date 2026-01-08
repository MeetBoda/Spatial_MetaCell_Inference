import warnings

warnings.simplefilter("ignore")
import copy
import argparse
import numpy as np
import scanpy as sc
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
)


def match(preds, targets):
    pred_num = preds.max() + 1
    mapping = {}
    for i in range(pred_num):
        pred_index = preds == i
        targets_index = targets[pred_index]
        most_common = np.bincount(targets_index).argmax()
        mapping[i] = most_common
    return mapping


def meta2origin(embedding, meta_ids):
    adata = sc.AnnData(embedding, dtype="float32")
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.louvain(adata, resolution=5.0)
    cluster_pred = adata.obs["louvain"].to_numpy().astype(int)
    original_cluster_pred = -np.ones_like(meta_ids, dtype=int)
    for i in range(embedding.shape[0]):
        original_cluster_pred[meta_ids == i] = cluster_pred[i]
    return original_cluster_pred


def main(args):
    original_adata = sc.read_h5ad(args.original_path)
    annotations = original_adata.obs[args.type_key].cat.codes.values
    metacell_adata = sc.read_h5ad(args.metacell_path)
    metacell_embedding = metacell_adata.obsm["X_pca_harmony"]
    metacell_id_adata = sc.read_h5ad(args.metacell_id_path)
    meta_ids = metacell_id_adata.obs["metacell"].values

    cluster_preds = meta2origin(metacell_embedding, meta_ids)
    cluster_preds_match = copy.deepcopy(cluster_preds)
    mapping = match(cluster_preds, annotations)
    for i in mapping.keys():
        cluster_preds_match[cluster_preds == i] = mapping[i]

    print("* AMI:", adjusted_mutual_info_score(annotations, cluster_preds_match))
    print("* ARI:", adjusted_rand_score(annotations, cluster_preds_match))
    print("* Homogeneity:", homogeneity_score(annotations, cluster_preds_match))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--original_path", type=str)
    parser.add_argument("--metacell_path", type=str)
    parser.add_argument("--metacell_id_path", type=str)
    parser.add_argument("--type_key", type=str, default="celltype")

    args = parser.parse_args()

    main(args)
