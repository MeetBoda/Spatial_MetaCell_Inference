import argparse
import scanpy as sc
from sklearn.metrics import homogeneity_score


def main(args):
    metacell_id_A_adata = sc.read_h5ad(args.metacell_id_A_path)
    meta_ids_A = metacell_id_A_adata.obs["metacell"].values
    metacell_id_B_adata = sc.read_h5ad(args.metacell_id_B_path)
    meta_ids_B = metacell_id_B_adata.obs["metacell"].values

    consistency = homogeneity_score(meta_ids_A, meta_ids_B)
    print("* Homogeneity Score:", consistency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--metacell_id_A_path", type=str)
    parser.add_argument("--metacell_id_B_path", type=str)

    args = parser.parse_args()

    main(args)
