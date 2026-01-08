Pre-requisite : We would need scRNA data with latent embedding (obtained from ENVI).

Step 1 : Run covet-eigenvalues-corrected.ipynb to generate the eigenvalues for the COVET matrix.
Step 1 : Execute concating_gene_with_eigenval.py file (inside the file change name of path accordingly to file taken in consideration)
Step 2 : Call MetaQ using new gene_eigenval concated data as RNA data and execute it.
Step 3 : Different h5ad files can be found in save folder
Step 4 : Different Plots can be found in figures folder
