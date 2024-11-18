#!/usr/bin/env python
# coding: utf-8

# # 1. import packages

# In[1]:


import argparse
import scAGDE
import scanpy as sc


# # 2. configurations

# In[2]:


parser = argparse.ArgumentParser(description='scAGDE')
parser.add_argument('--data', type=str, default='data/Forebrain.h5ad')
parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=8)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id to use.')
parser.add_argument('--seed', type=int, default=25, help='Random seed for repeat results')
parser.add_argument('--model', type=str, default=None, help='Load the trained model')
parser.add_argument('--verbose', "-v", action='store_false')
parser.add_argument('--outdir', '-o', type=str, default='output', help='Output path')
args = parser.parse_args(args=[])

scagde = scAGDE.Trainer(n_centroids=args.n_centroids, gpu=args.gpu, seed=args.seed, verbose=args.verbose, outdir=args.outdir)


# # 3. data preparation

# You can directly use the following code to read files in h5ad format or write one yourself.The `binary` parameter determines whether binarisation is performed (default is True).

# In[3]:


adata = scAGDE.utils.prepare_data(datapath=args.data,binary=True)
X = adata.X.toarray()


# # 4. run scAGDE

# We provide two ways of runnning scAGDE, as shown in `4.1` and `4.2`.

# ## 4.1 run scAGDE in 'step-by-step' style

# Firstly, a chromatin accessibility-based autoencoder is used to project the raw data into the low-dimensional space and search for neighbours for each cell in this space to constitute a cell graph.

# In[4]:


A_hat = scagde.CountModel(X)


# Meanwhile,the importance score of each peak is calculated based on the weights of the encoder module.Use the code below to plot the distribution of importance scores and choose the number of peaks to keep.

# In[5]:


scagde.plotPeakImportance()


# In[6]:


# Choose any moment after the scoring curve has flattened out, for example, 100000.
X, idx = scagde.peakSelect(X, topn=10000, return_idx=True)


# The graph learning module is utilised to obtain the final embedding as well as clustering results. `impute` determines whether or not to return the estimated true chromatin accessibility landscapes (default is True).

# In[7]:


cluster, embedding, impute = scagde.GraphModel(X, A_hat, impute=True)


# ## 4.2 run scAGDE in 'end-to-end' style

# Alternatively, you can simply omit the intermediate steps and run scAGDE in end-to-end way as below:

# In[8]:


# cluster, embedding, impute = scagde.fit(X, impute=True, topn=10000)


# ## clustering evaluation

# In[9]:


y = adata.obs["celltype"].astype("category").cat.codes.values
if y.min() > 0:
    y -= y.min()
res = scAGDE.utils.cluster_report(y, cluster.astype(float).astype(int))


# # 5. Integration with Scanpy

# In[10]:


adata.obsm["latent"] = embedding
adata.obs["cluster"] = cluster
adata.var["is_selected"] = 0
adata.var["is_selected"][idx] = 1
sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
sc.set_figure_params(dpi=80, figsize=(6, 6), fontsize=10)
sc.tl.umap(adata, min_dist=0.1)
color = [c for c in ["cluster", "celltype"] if c in adata.obs]
sc.pl.umap(adata, color=color, wspace=0.4, ncols=4, show=True)

