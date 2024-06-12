import argparse
from trainer import Trainer
from base.utils import *


if __name__ == '__main__':
    import scanpy as sc

    parser = argparse.ArgumentParser(description='GCN Encoder+MLP Decoder构成VGAE结构，加入DEC聚类')
    # 数据处理
    parser.add_argument('--data', type=str, default='datasets/Leukemia.h5ad')
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=6)
    # 训练
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=25, help='Random seed for repeat results')
    parser.add_argument('--model', type=str, default=None, help='Load the trained model')
    parser.add_argument('--verbose', "-v", action='store_false')
    # save
    parser.add_argument('--outdir', '-o', type=str, default='output', help='Output path')

    args = parser.parse_args()

    scagde = Trainer(n_centroids=args.n_centroids, gpu=args.gpu, seed=args.seed, verbose=args.verbose, outdir=args.outdir)

    # TODO 'step-by-step' style
    adata = prepare_data(datapath=args.data)
    X = adata.X.toarray()
    A_hat = scagde.CountModel(X)

    scagde.plotPeakImportance()
    X, idx = scagde.peakSelect(X, topn=10000, return_idx=True)

    cluster, embedding, impute = scagde.GraphModel(X, A_hat)

    # TODO 'end-to-end' style
    adata = prepare_data(datapath=args.data)
    X = adata.X.toarray()
    cluster, embedding, impute = scagde.fit(X, impute=True, topn=10000)

    # TODO evaluation
    print("\n ")
    y = adata.obs["celltype"].astype("category").cat.codes.values
    if y.min() > 0:
        y -= y.min()
    res = cluster_report(y, cluster.astype(float).astype(int))

    # TODO scanpy
    adata.obsm["latent"] = embedding
    adata.obs["cluster"] = cluster
    adata.var["is_selected"] = 0
    adata.var["is_selected"][idx] = 1
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    sc.settings.figdir = scagde.outdir
    sc.set_figure_params(dpi=80, figsize=(6, 6), fontsize=10)
    sc.tl.umap(adata, min_dist=0.1)
    color = [c for c in ["cluster", "celltype"] if c in adata.obs]
    sc.pl.umap(adata, color=color, save='.png', wspace=0.4, ncols=4, show=False)
