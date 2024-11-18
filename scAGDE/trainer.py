import argparse
import random
import warnings
from torch.backends import cudnn
from scAGDE.mclust import mclust_R
from scAGDE.utils import get_adj, cluster_report,prepare_data
import torch
from scAGDE.model import GraphEmbeddingModel, GraphEmbeddingModel_scale,\
    ChromatinAccessibilityAutoEncoder,ChromatinAccessibilityAutoEncoder_scale
import matplotlib.pyplot as plt
import os
import math
import numpy as np

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def estimate_optimal_k_louvain(data, max_resolution=1.0, step=0.1):
    import anndata
    import scanpy as sc
    from sklearn.metrics import silhouette_score
    best_k = 0
    best_score = -1
    best_resolution = 0.0
    adata = anndata.AnnData(data)
    # # # 进行 PCA 降维
    # sc.tl.pca(adata, n_comps=150)
    sc.pp.neighbors(adata)
    # # 遍历不同的分辨率来进行 Louvain 聚类
    resolutions = np.arange(0.1, max_resolution + step, step)
    for res in resolutions:
        sc.tl.leiden(adata, resolution=res)
        cluster_labels = adata.obs['leiden']
        if len(set(cluster_labels)) <= 1:
            continue
        # 计算轮廓系数来评估聚类质量
        score = silhouette_score(adata.X, cluster_labels.astype('int'))
        print(f'Resolution: {res}, Clusters: {len(set(cluster_labels))}, Silhouette Score: {score}')
        # 找到最好的轮廓系数
        if score > best_score:
            best_k = len(set(cluster_labels))
            best_score = score
            best_resolution = res
    print(f'Estimated Optimal Number of Clusters: {best_k} at Resolution: {best_resolution}')
    return best_k, best_resolution

class Trainer:
    def __init__(
        self, 
        adata, 
        outdir="./output/", 
        batch_key=None, 
        n_centroids=None, 
        seed=22,
        gpu="0",
        verbose=True,
        cluster_opt = True,
        save_epoch = None,
        early_stopping = False,
        lr1 = 0.0002,
        lr2 = 0.0002,
        max_iter1 = 5000,
        max_iter2 = 4000,
        pretrain_iter = 1500,
        weight_decay = 5e-4,
        patience = 50,
        clip = 0,
        dim_hidden_enc = [128],
        dim_hidden_dec = [],
        dim_latent = 10,
        ):
        """
            scAGDE: a single-cell chromatin accessibility model-based deep graph embedded learning method
        Args:
            adata (.h5ad file): .h5ad file path
            outdir (str, optional): output path. Defaults to "./output/".
            batch_key (str, optional): batch key in adata.obs.keys(). Defaults to "batch".
            n_centroids (int, optional): number of centroids in clustering layer. Defaults to None. if not specified, the estimation of cluster numbers will be activated.
        """        
        self.n_centroids = n_centroids
        self.seed = seed
        self.outdir = outdir
        self.verbose = verbose
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # TODO set device
        if gpu is None:
            self.device = 'cpu'
        else:
            if torch.cuda.is_available():  # cuda device
                self.device = torch.device('cuda:%s' % gpu)
            else:
                raise Exception("There is no cuda available on your device. Please check or modify parameter gpu='cpu'.")
        print("device used: %s\n" % self.device)
        # TODO Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        fix_seed(seed)
        self.peakImportance = None
        # TODO Set data        
        self.adata = adata
        self.batch_key = batch_key
        # TODO training configuration
        self.cluster_opt = cluster_opt
        self.save_epoch = save_epoch        
        self.early_stopping = early_stopping
        
        self.lr1 = lr1
        self.lr2 = lr2
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.pretrain_iter = pretrain_iter
        self.weight_decay = weight_decay
        self.patience = patience
        self.clip = clip
        self.dim_hidden_enc = dim_hidden_enc
        self.dim_hidden_dec = dim_hidden_dec
        self.dim_latent = dim_latent
        


    def fit(self, topn=10000,embed_init_key="latent_init",replace=False,selected_key="is_selected",impute_key="impute",cluster_key="cluster",embed_key="latent"):
        # TODO model initialization
        self.CountModel(embed_init_key=embed_init_key)
        self.peakSelect(topn=topn,replace=replace,selected_key=selected_key)
        return self.GraphModel(impute_key=impute_key,cluster_key=cluster_key,embed_key=embed_key)

    def CountModel(self,embed_init_key="latent_init"):
        fix_seed(self.seed)
        # TODO data preparation
        data = self.adata.X
        if not isinstance(data,np.ndarray): data = data.toarray()
        # TODO model configuration
        input_dim = data.shape[1]
        dims = [input_dim, 10, [128], []]
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(self.adata.shape[0], input_dim, self.n_centroids))
        model = ChromatinAccessibilityAutoEncoder(dims, device=self.device)
        model.to(self.device)
        print('\n## Training CountModel ##')
        model.fit(X=data, lr=self.lr1, weight_decay=self.weight_decay, max_iter=self.max_iter1, verbose=self.verbose, outdir=self.outdir,save_epoch=self.save_epoch,clip=self.clip)
        weight_tensor = model.encoder.hidden[0].weight.detach().cpu().data
        var_tensor = torch.std(weight_tensor, 0).numpy()
        self.peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
        torch.cuda.empty_cache()
        with torch.no_grad():
            z = model.encodeBatch(data)
        self.adata.obsm[embed_init_key] = z
        self.embed_init_key = embed_init_key
        # TODO estimation
        if self.n_centroids is None:
            n_centroids,_ = estimate_optimal_k_louvain(z)
            if n_centroids < 2:
                n_centroids = 2
            self.estimates = n_centroids
        else:
            self.estimates = self.n_centroids

    def peakSelect(self, topn=10000, replace=False, selected_key="is_selected"):
        """
        select peaks by peak importance scores
        Args:
            topn (int, optional): The number of peaks to be obtained. Defaults to 10000.
            replace (bool, optional): Inplace subset to top-important peaks if `True` otherwise merely indicate these peaks. Defaults to False.
            selected_key (str, optional): if replace=False, the selected peaks are labeled 1 in adata.var[selected_key]. Defaults to `is_selected`.
        Returns:
            
        """        
        if self.peakImportance is None:
            raise RuntimeError("[scAGDE]: Trying to query peak importance scores from an untrained model. Please train the model first.")
        if len(self.peakImportance) != self.adata.shape[1]:
            raise RuntimeError("[scAGDE]: Size mismatched! It seems the dataset used for training is not identical to this one.")
        if topn >= self.adata.shape[1]:
            warnings.warn("[scAGDE]: The number of peaks to select exceeds the total number of peaks, you can reduce the `topn` to select fewer peaks.")
        idx = np.argsort(self.peakImportance)[::-1][:topn]
        if replace:
            self.adata = self.adata[:,idx]
        else:
            self.adata.var[selected_key] = 0
            self.adata.var.iloc[idx, self.adata.var.columns.get_loc(selected_key)] = 1
        self.replace = replace
        self.selected_key = selected_key



    def GraphModel(self, impute_key=None, cluster_key="cluster", embed_key="latent"):
        fix_seed(self.seed)        
        outdir = self.outdir
        # TODO data preparation
        if (self.selected_key not in self.adata.var.keys()) and (self.replace is False):
            data = self.adata.X
        elif self.replace:
            data = self.adata.X
        else:
            data = self.adata[:,self.adata.var[self.selected_key] == 1].X
        if not isinstance(data,np.ndarray): data = data.toarray()
        input_dim = data.shape[1]
        # TODO knn graph
        print('\n## Constructing Cell Graph ##')
        z = self.adata.obsm[self.embed_init_key]
        _, adj_n = get_adj(z, pca=None)
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(self.adata.shape[0], input_dim, self.estimates))
        model = GraphEmbeddingModel([input_dim, self.dim_latent, self.dim_hidden_enc,self.dim_hidden_dec], n_centroids=self.estimates, device=self.device,cluster_opt=self.cluster_opt)
        model.to(self.device)
        print('\n## Training GraphModel ##')
        model.fit(
            adj_n=adj_n, X=data,
            lr=self.lr2, weight_decay=self.weight_decay, max_iter=self.max_iter2, pre_iter=self.pretrain_iter, outdir=outdir, verbose=self.verbose,
            update_interval=100,
            patience=self.patience,
            clip=self.clip,
            early_stopping=self.early_stopping
        )
        torch.cuda.empty_cache()
        # TODO output
        if impute_key is not None:
            z, x_bar = model.encodeBatch(adj_n, data, impute=True)
            self.adata.obsm[impute_key] = x_bar
        else:
            z = model.encodeBatch(adj_n, data, impute=False)
        self.adata.obsm[embed_key] = z
        if cluster_key is not None:
            cluster = mclust_R(z, self.estimates).astype(int).astype(str)
            self.adata.obs[cluster_key] = cluster
        self.embed_key = embed_key
        self.impute_key = impute_key
        self.cluster_key = cluster_key
        return self.adata

    def plotPeakImportance(self):
        if self.peakImportance is None:
            warnings.warn(
                "Trying to query peak importance scores from an untrained model. Please train the model first.")
        else:
            sorted_data = sorted(self.peakImportance, reverse=True)
            x = np.arange(1, len(sorted_data) + 1)
            # 绘制图表
            plt.figure(figsize=(10, 6))
            plt.plot(x, sorted_data, marker='o', linestyle='-', color='black')
            # 添加标题和标签
            plt.title('Sorted PeakImportance Plot')
            plt.xlabel('Number')
            plt.ylabel('score')
            # 显示网格
            plt.grid(True)
            # 显示图表
            plt.show()




class Trainer_scale(Trainer):

    def __init__(
        self,
        adata,
        outdir="./output/",
        batch_key=None, 
        n_centroids=None,
        seed=22, 
        gpu="0",
        verbose=True,
        cluster_opt = True,
        save_epoch = None,
        early_stopping = False,
        lr1 = 0.0002,
        lr2 = 0.0002,
        max_iter1 = 200,
        max_iter2 = 300,
        pretrain_iter = 200,
        weight_decay = 7e-4,
        patience = 50,
        clip = 0,
        dim_hidden_enc = [128,128],
        dim_hidden_dec = [128],
        dim_latent = 10,
        batch_size_ae=128,
        batch_size_gae=320,
        centroids_init="kmeans",
        ):        
        super().__init__(adata, outdir, batch_key, n_centroids, seed, gpu, verbose,cluster_opt,save_epoch,early_stopping)
        self.lr1 = lr1
        self.lr2 = lr2
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.pretrain_iter = pretrain_iter
        self.weight_decay = weight_decay
        self.patience = patience
        self.clip = clip
        self.dim_hidden_enc = dim_hidden_enc
        self.dim_hidden_dec = dim_hidden_dec
        self.dim_latent = dim_latent
        
        self.batch_size_ae = batch_size_ae
        self.batch_size_gae = batch_size_gae
        self.centroids_init = centroids_init


    def fit(self, topn=10000,embed_init_key="latent_init",replace=False,selected_key="is_selected",impute_key="impute",cluster_key="cluster",embed_key="latent"):
        # TODO model initialization
        self.CountModel(embed_init_key=embed_init_key)
        self.peakSelect(topn=topn,replace=replace,selected_key=selected_key)
        return self.GraphModel(impute_key=impute_key,cluster_key=cluster_key,embed_key=embed_key)

    def CountModel(self,embed_init_key="latent_init"):
        fix_seed(self.seed)
        # TODO data preparation
        data = self.adata.X
        if not isinstance(data,np.ndarray): data = data.toarray()
        # TODO model configuration
        input_dim = data.shape[1]
        dims = [input_dim, 10, [128], []]
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(data.shape[0], input_dim, self.n_centroids))
        num_batch = int(math.ceil(1.0 * len(self.adata) / self.batch_size_ae))
        batch = []
        for i in range(num_batch):
            batch.append(data[i*self.batch_size_ae:min((i+1)*self.batch_size_ae,len(self.adata))])
        data = batch
        del batch
        model = ChromatinAccessibilityAutoEncoder_scale([input_dim, self.dim_latent, self.dim_hidden_enc,self.dim_hidden_dec], device=self.device)
        model.to(self.device)
        print('\n## Training CountModel ##')
        model.fit(X=data, lr=self.lr1, weight_decay=self.weight_decay, max_iter=self.max_iter1, verbose=self.verbose, outdir=self.outdir,save_epoch=self.save_epoch,clip=self.clip)
        weight_tensor = model.encoder.hidden[0].weight.detach().cpu().data
        var_tensor = torch.std(weight_tensor, 0).numpy()
        self.peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
        torch.cuda.empty_cache()
        with torch.no_grad():
            z = model.encodeBatch(data)
        self.adata.obsm[embed_init_key] = z
        self.embed_init_key = embed_init_key
        # TODO estimation
        if self.n_centroids is None:
            n_centroids,_ = estimate_optimal_k_louvain(z)
            if n_centroids < 2:
                n_centroids = 2
            self.estimates = n_centroids
        else:
            self.estimates = self.n_centroids


    def GraphModel(self, impute_key=None, cluster_key="cluster", embed_key="latent"):
        fix_seed(self.seed)        
        outdir = self.outdir
        # TODO data preparation
        if (self.selected_key not in self.adata.var.keys()) and (self.replace is False):
            data = self.adata.X
        if self.replace:
            data = self.adata.X
        else:
            data = self.adata[:,self.adata.var[self.selected_key] == 1].X
        if not isinstance(data,np.ndarray): data = data.toarray()
        input_dim = data.shape[1]
        # TODO knn graph
        print('\n## Constructing Cell Graph ##')
        num_batch = int(math.ceil(1.0 * len(self.adata) / self.batch_size_gae))
        adj_n = []
        batch = []
        z = self.adata.obsm[self.embed_init_key]
        for i in range(num_batch):
            z_batch = z[i*self.batch_size_gae:min((i+1)*self.batch_size_gae,len(z))]
            X_batch = data[i*self.batch_size_gae:min((i+1)*self.batch_size_gae,len(z))]
            _, adj_n_batch = get_adj(z_batch, pca=None,k=3)
            adj_n.append(adj_n_batch)
            batch.append(X_batch)
        data = batch
        del batch
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(self.adata.shape[0], input_dim, self.estimates))
        model = GraphEmbeddingModel_scale([input_dim, self.dim_latent, self.dim_hidden_enc,self.dim_hidden_dec], n_centroids=self.estimates, device=self.device,cluster_opt=self.cluster_opt)
        model.to(self.device)
        print('\n## Training GraphModel ##')
        model.fit(
            adj_n=adj_n, X=data,
            lr=self.lr2, weight_decay=self.weight_decay, max_iter=self.max_iter2, pre_iter=self.pretrain_iter, outdir=outdir, verbose=self.verbose,
            update_interval=1,
            patience=self.patience,
            clip=self.clip,
            early_stopping=self.early_stopping,
            centroids_init=self.centroids_init
        )
        torch.cuda.empty_cache()
        # TODO output
        if impute_key is not None:
            z, x_bar = model.encodeBatch(adj_n, data, impute=True)
            self.adata.obsm[impute_key] = x_bar
        else:
            z = model.encodeBatch(adj_n, data, impute=False)
        self.adata.obsm[embed_key] = z
        if cluster_key is not None:
            cluster = mclust_R(z, self.estimates).astype(int).astype(str)
            self.adata.obs[cluster_key] = cluster
        self.embed_key = embed_key
        self.impute_key = impute_key
        self.cluster_key = cluster_key
        return self.adata

