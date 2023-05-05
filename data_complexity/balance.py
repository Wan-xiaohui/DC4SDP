import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


"""## 2.6 Class Imbalance Measures

### 2.6.1 Entropy of class proportions (C1)
"""


def ft_C1(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        pi = cls_n_ex[i]/n
        summation = summation + pi * math.log(pi)
    aux = 1 + summation / math.log(nc)
    return aux


"""### 2.6.2 Imbalance ratio (C2)"""


def ft_C2(cls_n_ex: np.ndarray) -> float:
    nc = len(cls_n_ex)
    n = sum(cls_n_ex)
    summation = 0
    for i in range(nc):
        summation = summation + cls_n_ex[i]/(n - cls_n_ex[i])
    aux = ((nc - 1)/nc) * summation
    aux = 1 - (1/aux)
    return aux


"""###  Supplement: Bayes Imbalance Impact Index (BI3)"""
""" Reference: https://github.com/jasonyanglu/BI3 """


def ft_IBI3_and_BI3(data, label, k=5):
    """
    :param data: np.ndarray
    :param label: np.ndarray
    :param k: the recommended value is 5
    :return:
    """
    pos_num = sum(label == 1)
    neg_num = sum(label == 0)
    pos_idx = np.nonzero(label == 1)
    neg_idx = np.nonzero(label == 0)
    pos_data = data[pos_idx]
    rr = neg_num / pos_num

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    distances, knn_idx = nbrs.kneighbors(pos_data)

    p2 = np.zeros(pos_num)
    p2old = np.zeros(pos_num)
    knn_idx = np.delete(knn_idx, 0, 1)

    for i in range(pos_num):
        p2[i] = np.intersect1d(knn_idx[i], neg_idx).size / k
        p2old[i] = p2[i]

        if p2[i] == 1:
            dist = pairwise_distances(pos_data[i].reshape(1, -1), data).reshape(-1)
            sort_idx = np.argsort(dist)
            nearest_pos = np.nonzero(label[sort_idx] == 1)[0][1]
            p2[i] = (nearest_pos - 1) / nearest_pos

    p1 = 1 - p2

    px = (rr * p1 / (p2 + rr * p1) - p1)

    pm = np.mean(px)

    return px, pm
