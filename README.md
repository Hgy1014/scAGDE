# scAGDE

[![PyPI badge](https://badge.fury.io/py/scAGDE.svg)](https://pypi.org/project/scMGCA/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11609252.svg)](https://doi.org/10.5281/zenodo.11609252)
![logo](https://github.com/Hgy1014/scAGDE/assets/64194550/867c48cc-c777-4a08-9886-eb6fdb214cc5)

`scAGDE` is a Python implementation for a novel single-cell chromatin accessibility model-based deep graph representation learning method that simultaneously learns feature representation and
clustering through explicit modeling of single-cell ATAC-seq data generation.
- [Briefly](#Briefly)
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [Data Availability](#data-availability)
- [License](#license)

# Briefly
Single-cell ATAC-seq technology has significantly advanced our understanding of cellular heterogeneity by enabling the exploration
of epigenetic landscapes and regulatory elements at the single-cell level. A major challenge in analyzing high-throughput single-cell
ATAC-seq data is its inherently low copy number, leading to data sparsity and high dimensionality, significantly limiting the elucidation and characterization of gene regulatory elements. To address these limitations, we developed scAGDE, a novel single-cell chromatin accessibility model-based deep graph representation learning method that simultaneously learns feature representation and
clustering through explicit modeling of single-cell ATAC-seq data generation. scAGDE first leverages a chromatin accessibility-based
autoencoder, which is designed to identify key patterns in single-cell ATAC-seq data, eliminate less relevant peaks, and construct a cell
graph to elucidate the topological connections among individual cells. After that, scAGDE integrates a Graph Convolutional Network
(GCN) as an encoder to extract essential structural information from both the ATAC-seq count matrix and the cell graph, coupled
with a Bernoulli-based decoder to characterize the global probabilistic structure of the data. Additionally, the graph embedding
process independently generates soft labels that guide self-supervised deep clustering, which is characterized by its iterative refinement of results.
# Overview
Overview of the scAGDE framework. (a) A summary graphical illustration of scAGDE workflow. scAGDE takes as input the binary cell-by-peak matrix first into
a chromatin accessibility-based autoencoder and then performs the graph embedding learning. (b) The chromatin accessibility-based autoencoder maps data into latent
space, where each individual cell connects its nearest cell as neighbours to construct a cell graph. The variation of encoder’s weights can be translated to importance score
of peaks for peak selection procedure. (c) The well-prepared cell graph and filtered data are simultaneously handled by a two-layer GCN encoder (i) and mapped into the
latent space (ii). On the one hand, the latent embedding serves as input to dual decoders (iii), which include a graph decoder module to reconstruct from embedding, and a
Bernoulli-based decoder module to estimate the probability of a peak being accessible, which are estimates of the true chromatin landscape in each cell. On the other hand,
the dual clustering optimizations are introduced (iv), where a network of cluster layer, which is initialized by K-means results on the embedding, infers soft clustering label.
The target distribution and one-hot pseudo label are sequentially calculated and used for label prediction loss and distribution alignment loss. (d) scAGDE facilitates critical
downstream applications of clustering, visualization, imputation, enrichment analysis and discovery of regulators.
![framework](https://github.com/Hgy1014/scAGDE/assets/64194550/79b02f20-7bde-4849-abc2-89d5bae66ce3)

# System Requirements
## Hardware requirements
`scMGCA` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 18.04

### Python Dependencies
`scMGCA` mainly depends on the Python scientific stack.
```
numpy
scipy
tensorflow
scikit-learn
pandas
scanpy
anndata
```
For specific setting, please see <a href="https://github.com/Philyzh8/scMGCA/blob/master/requirements.txt">requirement</a>.

# Installation Guide:

### Install from PyPi

```
$ conda create -n scMGCA_env python=3.6.8
$ conda activate scMGCA_env
$ pip install -r requirements.txt
$ pip install scMGCA
```

# Usage
`scMGCA` is a deep graph embedding learning method for single-cell clustering, which can be used to:
+ Single-cell data clustering. The example can be seen in the <a href="https://github.com/Philyzh8/scMGCA/blob/master/tutorial/demo.py">demo.py</a>.
+ Correct the batch effect of data from different scRNA-seq protocols. The example can be seen in the <a href="https://github.com/Philyzh8/scMGCA/blob/master/tutorial/demo_batch.py">demo_batch.py</a>.
+ Analysis of the mouse brain data with 1.3 million cells. The example can be seen in the <a href="https://github.com/Philyzh8/scMGCA/blob/master/tutorial/demo_scale.py">demo_scale.py</a>.
+ Provide an automatic hyperparameter search algorithm. The example can be seen in the <a href="https://github.com/Philyzh8/scMGCA/blob/master/tutorial/demo_para.py">demo_para.py</a>.

We give users some suggestions for running in the <a href="https://github.com/Philyzh8/scMGCA/blob/master/tutorial/tutorial.md">tutorial.md</a>.


# Data Availability

The real data sets we used can be download in <a href="https://doi.org/10.5281/zenodo.7475687">data</a>.

# License

This project is covered under the **MIT License**.

# Citation

```
@article{yu2023topological,
  title={Topological identification and interpretation for single-cell gene regulation elucidation across multiple platforms using scMGCA},
  author={Yu, Zhuohan and Su, Yanchi and Lu, Yifu and Yang, Yuning and Wang, Fuzhou and Zhang, Shixiong and Chang, Yi and Wong, Ka-Chun and Li, Xiangtao},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={400},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

