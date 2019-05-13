""" finds the knn based on dot-product """
import numpy as np


def simple_numpy_knn(queries: np.ndarray, vectors: np.ndarray, k: int, return_scores=False):
    scores = queries.dot(vectors.T)  # (num_q, num_v) - each row corresponding to scores
    k = min(scores.size, k)
    if return_scores:
        return np.argpartition(-scores, kth=list(range(k)), axis=1)[:, :k], \
               -np.partition(-scores, kth=list(range(k)), axis=1)[:, :k]
    return np.argpartition(-scores, kth=list(range(k)), axis=1)[:, :k]


def numpy_global_knn(queries: np.ndarray, vectors: np.ndarray, k: int):
    scores = queries.dot(vectors.T)
    k = min(scores.size, k)
    return np.stack(np.unravel_index(np.argpartition(-scores, kth=list(range(k)), axis=None), scores.shape), axis=1)[:k]
