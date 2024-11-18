import scanpy as sc
import numpy as np
import scipy
from scipy.sparse import issparse
from scipy import sparse as sp
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA


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


def binarization(imputed, raw):
    return scipy.sparse.csr_matrix((imputed.T > raw.mean(1).T).T & (imputed>raw.mean(0))).astype(np.int8)

def prepare_data(datapath, binary=True):
    print("Loading dataset from:%s" % datapath)
    adata = sc.read_h5ad(datapath)
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    if binary:
        adata.X[adata.X > 1] = 1
    return adata


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def dopca(X, dim=10):
    if dim <= 1:
        dim = float(dim)
    else:
        dim = int(dim)
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def get_adj(count, k=15, pca=50, mode="connectivity", metric="euclidean"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric=metric, include_self=True)
    adj = A.A
    adj_n = norm_adj(A).A
    return adj, adj_n


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()




#### Data Processing ####
def gene_filter_(data, X=6):
    total_cells = data.shape[1]
    count_1 = data[data >= 1].count(axis=1)
    count_2 = data[data > 0].count(axis=1)

    genelist_1 = count_1[count_1 > 0.01 * X * total_cells].index
    genelist_2 = count_2[count_2 < 0.01 * (100 - X) * total_cells].index
    genelist = set(genelist_1) & set(genelist_2)
    data = data.loc[genelist]
    return data


def sort_by_mad(data, axis=0):
    genes = data.mad(axis=axis).sort_values(ascending=False).index
    if axis == 0:
        data = data.loc[:, genes]
    else:
        data = data.loc[genes]
    return data


#### scATAC Preprocessing ####
def peak_filter(data, x=10, n_reads=2):
    count = data[data >= n_reads].count(1)
    index = count[count >= x].index
    data = data.loc[index]
    return data


def cell_filter(data):
    thresh = data.shape[0] / 50
    # thresh = min(min_peaks, data.shape[0]/50)
    data = data.loc[:, data.sum(0) > thresh]
    return data


def sample_filter(data, x=10, n_reads=2):
    data = peak_filter(data, x, n_reads)
    data = cell_filter(data)
    return data


#### Metrics ####
def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """

    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        # for (i, j) in index:
        for (i, j) in zip(index[0], index[1]):
            y_[np.where(y_pred == i)] = j
        return y_

    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    #     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)


def cluster_report(ref, pred, x=None):
    print("\n## Clustering Evaluation Report ##")
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('# Confusion matrix: #')
    print(cm)
    ari = adjusted_rand_score(ref, pred)
    nmi = normalized_mutual_info_score(ref, pred)
    f1 = f1_score(ref, pred, average='micro')
    print('# Metric values: #')
    print("Adjusted Rand Index: {:.4f}".format(ari))
    print("Normalized Mutual Info: {:.4f}".format(nmi))
    print("F1 score: {:.4f}".format(f1))
    res = locals()
    res.pop("ref")
    res.pop("pred")
    return res


def binarization(imputed, raw):
    return scipy.sparse.csr_matrix((imputed.T > raw.mean(1).T).T & (imputed > raw.mean(0))).astype(np.int8)
