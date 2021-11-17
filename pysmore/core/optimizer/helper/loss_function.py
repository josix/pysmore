"""
Helper functions for getting loss in parallel.
"""
from typing import Tuple

import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "threadsafe"


@njit(parallel=False)
def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.
    """
    return sum([v1[i] * v2[i] for i in prange(v1.shape[0])])


@njit(parallel=False)
def compute_dot_product_update(
    embeddings: np.ndarray,
    training_edges: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes total loss of all edges.
    """
    update_embedding = np.zeros(embeddings.shape)
    loss = 0.0
    for edge_idx in prange(training_edges.shape[0]):
        source_node = int(training_edges[edge_idx, 0])
        target_node = int(training_edges[edge_idx, 1])
        weight = training_edges[edge_idx, 2]
        gradient = weight - dot_product(
            embeddings[source_node], embeddings[target_node]
        )
        loss += gradient
        update_embedding[source_node] += gradient * embeddings[source_node]
        update_embedding[target_node] += gradient * embeddings[target_node]
    return (update_embedding, loss)
