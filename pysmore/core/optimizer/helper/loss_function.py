"""
Helper functions for getting loss in parallel.
"""
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "threadsafe"


@njit(parallel=True)
def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.
    """
    return sum([v1[i] * v2[i] for i in prange(v1.shape[0])])


@njit(parallel=True)
def compute_raw_dot_product_loss(
    embeddings: np.ndarray,
    training_edges: np.ndarray,
) -> np.ndarray:
    """
    Computes total loss of all edges.
    """
    loss = np.zeros(embeddings.shape)
    for edge_idx in prange(training_edges.shape[0]):
        source_node = int(training_edges[edge_idx, 0])
        target_node = int(training_edges[edge_idx, 1])
        weight = training_edges[edge_idx, 2]
        gradient = weight - dot_product(
            embeddings[source_node], embeddings[target_node]
        )
        loss[source_node] += gradient * embeddings[source_node]
        loss[target_node] += gradient * embeddings[target_node]
    return loss
