# 🧬 Inference of Spatial Metacells by Integrating Single-Cell and Spatial Transcriptomics

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Single-Cell](https://img.shields.io/badge/Scanpy-AnnData-brightgreen.svg)](https://scanpy.readthedocs.io/)
[![Institute](https://img.shields.io/badge/IIT%20Kanpur-CSE%20Department-darkblue.svg)](https://www.iitk.ac.in/cse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research Project — Department of Computer Science & Engineering, IIT Kanpur**  
> **Authors:** Arghyadeep Saha, Meetkumar Boda, Siddharth Banerjee, Thanush A A  
> **Affiliation:** Indian Institute of Technology Kanpur (IITK)

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Core Theoretical Background](#-core-theoretical-background)
  - [1. COVET (Covariance Environment) Formulation](#1-covet-covariance-environment-formulation)
  - [2. ENVI (Environmental Variational Inference)](#2-envi-environmental-variational-inference)
  - [3. MetaQ (Single-Cell Quantization)](#3-metaq-single-cell-quantization)
- [Datasets](#-datasets)
- [Methodologies (All 4 Integration Approaches)](#-methodologies-all-4-integration-approaches)
  - [Method 1: Creating and Splitting MetaCells (Post-MetaQ Splitting)](#method-1-creating-and-splitting-metacells-post-metaq-splitting)
  - [Method 2: Appending Eigenvalues of the COVET Matrix](#method-2-appending-eigenvalues-of-the-covet-matrix)
  - [Method 3: Direct Integration via ENVI Shared Latent Embeddings](#method-3-direct-integration-via-envi-shared-latent-embeddings)
  - [Method 4: Spatial Clustering via COVET Graph + Cluster-Wise MetaQ](#method-4-spatial-clustering-via-covet-graph--cluster-wise-metaq)
- [Experimental Results & Evaluation](#-experimental-results--evaluation)
  - [Quantitative Metrics](#quantitative-metrics)
  - [UMAP Visualizations & Biological Insights](#umap-visualizations--biological-insights)
- [Project Architecture & Repository Layout](#-project-architecture--repository-layout)
- [Getting Started & Reproducibility](#-getting-started--reproducibility)
- [References](#-references)
- [Contributors](#-contributors)

---

## 🔬 Executive Summary

Metacell partitioning algorithms aggregate individual, noisy single-cell transcriptomes into distinct, homogeneous, high-granularity cellular states (*metacells*). However, traditional metacell algorithms (e.g., standard MetaQ, MetaCell) operate purely in gene expression space, ignoring physical tissue coordinates. Consequently, cells that are transcriptomically identical but located in vastly different anatomical microenvironments (niches) are incorrectly collapsed into the same metacell.

This project introduces and benchmarks **4 novel computational pipelines** to infer **Spatial Metacells** by jointly modeling single-cell RNA sequencing (**scRNA-seq**) and Spatial Transcriptomics (**ST / MERFISH**) data via the integration of **ENVI** and **MetaQ**.


---

## 🎯 Motivation & Problem Statement

* **Single-Cell RNA-seq:** Measures high gene throughput genome-wide (~30,000+ genes) but loses all in-situ spatial context due to tissue dissociation.
* **Spatial Transcriptomics (MERFISH):** Preserves spatial $(x, y)$ coordinates in tissue slices but is restricted to targeted gene panels (~254 genes).
* **The Goal:** Construct spatial metacells that preserve **both full transcriptomic state fidelity** and **local spatial microenvironment niche homogeneity**.

---

## 📐 Core Theoretical Background

### 1. COVET (Covariance Environment) Formulation
To represent the spatial niche of cell $i$, we define its cellular neighborhood as the $k$-nearest spatial neighbors:
$$\text{Niche}(i) = \{j \mid j \in k\text{-NN}(i)\}$$

The gene expression vectors of these neighbors form a niche matrix $E_i = \{Y_{ij} \in \mathbb{R}^g \mid j \in k\text{-NN}(i)\}$. The shifted niche covariance matrix $\Sigma_i$ is computed relative to the global dataset mean expression $\bar{X}$:
$$\Sigma_i = \text{ShiftCov}(E_i) = \frac{1}{k} (E_i - \bar{X})^T (E_i - \bar{X})$$
This ensures all cell niches are projected onto a shared reference frame, producing symmetric, positive semi-definite (PSD) covariance matrices.

### 2. ENVI (Environmental Variational Inference)
ENVI employs a conditional Variational Autoencoder (cVAE) with an auxiliary modality indicator neuron $c \in \{0, 1\}$ to project both modalities into a shared latent distribution $l \sim \mathcal{N}(\mu_l, \Sigma_l)$:
* **Decoder 1:** Reconstructs genome-wide expression $X$.
* **Decoder 2:** Reconstructs the local spatial covariance matrix $\Sigma_i$ (COVET).

### 3. MetaQ (Single-Cell Vector Quantization)
MetaQ applies vector quantization via a one-encoder, two-decoder deep learning framework to construct compact codebook representations ($c_j$):
$$\mathcal{L}_{\text{MetaQ}} = \sum_i \left( \|x_i - \hat{x}_i\|^2 + \lambda \|x_i - \hat{x}_i^{\text{meta}}\|^2 \right) + \beta \cdot \text{Reg}$$
Where $c_j = \frac{1}{|S_j|} \sum_{i \in S_j} z_i$ represents the metacell embedding vector.

---

## 📊 Datasets

Experiments were conducted on paired single-cell and spatial datasets from the **Mouse Motor Cortex**:

| Dataset | Modality / Technology | Cell / Spot Count | Gene Count | Spatial Coordinates $(x,y)$ |
| :--- | :--- | :--- | :--- | :--- |
| **`sc_data`** | Single-cell RNA-seq (scRNA-seq) | **71,183** cells | **30,618** genes | ❌ No |
| **`st_data`** | MERFISH Spatial Transcriptomics | **276,556** spots | **254** genes | ✅ Yes |

*Data source: Pe'er Lab AWS Open Access Bucket.*

---

## ⚙️ Methodologies (All 4 Integration Approaches)

```mermaid
graph TD
    subgraph Data Inputs
        SC[Single-Cell RNA-seq]
        ST[Spatial MERFISH]
    end

    SC & ST --> ENVI[ENVI Joint Learning]

    %% Method 1
    SC --> M1_MetaQ[MetaQ Baseline] --> M1_MC[Transcriptomic Metacells]
    ENVI --> M1_Latent[ENVI Latent Embeddings]
    M1_MC & M1_Latent --> M1_Kmeans[K-Means Spatial Splitting] --> Out1[Method 1: Split Spatial Metacells]

    %% Method 2
    ENVI --> M2_COVET[COVET Matrices 25x25] --> M2_Eig[Eigendecomposition / 25 Eigenvalues]
    SC & M2_Eig --> M2_Concat[Concatenate Features] --> M2_MetaQ[MetaQ Quantization] --> Out2[Method 2: Eigenvalue Metacells]

    %% Method 3
    ENVI --> M3_Latent[Joint Latent Embeddings] --> M3_MetaQ[MetaQ on Latent Space] --> Out3[Method 3: Shared Latent Metacells]

    %% Method 4
    ENVI --> M4_COVET[Single-Cell COVET] --> M4_Graph[KNN Graph via Approx. Optimal Transport]
    M4_Graph --> M4_Leiden[Leiden Spatial Clustering] --> M4_Clusters[Niche Sub-clusters]
    M4_Clusters --> M4_MetaQ[Cluster-wise MetaQ Inference] --> Out4[Method 4: COVET Clustered Metacells]
