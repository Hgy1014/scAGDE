import argparse
import random
import warnings
from torch.backends import cudnn
from scAGDE.mclust import mclust_R
from scAGDE.utils import get_adj, cluster_report,prepare_data
import torch
from scAGDE.model import GraphEmbeddingModel, ChromatinAccessibilityAutoEncoder
import matplotlib.pyplot as plt
import os
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


class Trainer:
    def __init__(self, n_centroids, gpu, seed, verbose=True, outdir="output/"):
        self.n_centroids = n_centroids
        self.gpu = gpu
        self.seed = seed
        self.verbose = verbose
        self.outdir = outdir
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # TODO device configuration
        if torch.cuda.is_available():  # cuda device
            self.device = torch.device('cuda:%d' % gpu)
        else:
            self.device = 'cpu'
        print("device used: %s\n" % self.device)
        # TODO Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        fix_seed(seed)
        self.peakImportance = None

    def fit(self, X, topn=10000, impute=True):
        A = self.CountModel(X)
        X = self.peakSelect(X, topn=topn, return_idx=False)
        return self.GraphModel(X, A, impute=impute)

    def GraphModel(self, X, A, impute=True):
        fix_seed(self.seed)
        outdir = self.outdir
        input_dim = X.shape[1]
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(X.shape[0], input_dim, self.n_centroids))
        model = GraphEmbeddingModel([input_dim, 10, [128], []], n_centroids=self.n_centroids, device=self.device)
        model.to(self.device)
        print('\n## Training GraphModel ##')
        model.fit(
            adj_n=A, X=X,
            lr=0.0002, weight_decay=5e-4, max_iter=4, pre_iter=1, outdir=outdir, verbose=self.verbose,
            update_interval=100
        )
        torch.cuda.empty_cache()
        # todo cluster
        with torch.no_grad():
            adj_n = torch.from_numpy(A).float().to(self.device)
            if impute:
                z, x_bar = model.encodeBatch(adj_n, X, impute)
            else:
                z = model.encodeBatch(adj_n, X, impute)
        cluster = mclust_R(z, self.n_centroids).astype(int).astype(str)
        if impute:
            return cluster, z, x_bar
        else:
            return cluster, z

    def CountModel(self, X):
        fix_seed(self.seed)
        # TODO model configuration
        input_dim = X.shape[1]
        dims = [input_dim, 10, [128], []]
        print('Cell number: {}\nPeak number: {}\nn_centroids: {}\n'.format(X.shape[0], input_dim, self.n_centroids))
        # VAE
        model = ChromatinAccessibilityAutoEncoder(dims, n_centroids=self.n_centroids, device=self.device)
        model.to(self.device)
        print('\n## Training CountModel ##')
        model.fit(X=X, lr=0.0002, weight_decay=5e-4, max_iter=5, verbose=self.verbose, outdir=self.outdir)
        weight_tensor = model.encoder.hidden[0].weight.detach().cpu().data
        var_tensor = torch.std(weight_tensor, 0).numpy()
        self.peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
        torch.cuda.empty_cache()
        # TODO knn graph
        with torch.no_grad():
            z = model.encodeBatch(X)
        print('\n## Constructing Cell Graph ##')
        _, adj_n = get_adj(z, pca=None)
        return adj_n

    def peakSelect(self, X, topn=10000, return_idx=False):
        if self.peakImportance is None:
            raise RuntimeError("[scAGDE]: Trying to query peak importance scores from an untrained model. Please train the model first.")
        if len(self.peakImportance) != X.shape[1]:
            raise RuntimeError("[scAGDE]: Size mismatched! It seems the dataset used for training is not identical to this one.")
        if topn >= X.shape[1]:
            warnings.warn("[scAGDE]: The number of peaks to select exceeds the total number of peaks, you can reduce the `topn` to select fewer peaks.")
        idx = np.argsort(self.peakImportance)[::-1][:topn]
        if return_idx:
            return X[:, idx], idx
        else:
            return X[:, idx]

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

