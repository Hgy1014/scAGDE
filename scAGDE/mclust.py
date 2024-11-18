import numpy as np
# from rpy2.robjects.packages import importr


def mclust_R(data, num_cluster, modelNames='EEE', random_seed=18):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r['options'](warn=-1)
    # importr('mclust')
    robjects.r.library("mclust")
    # robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    robjects.r['mclust.options'](subset=3000)
    robjects.r['options'](warn=-1)
    robjects.r['mclust.options'](warn=-1)
    try:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, modelNames, verbose=False)
        mclust_res = np.array(res[-2])

    except TypeError:
        mclust_res = np.ones(len(data))
        mclust_res[0] = 0

    return mclust_res
