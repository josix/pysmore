"""
Pair optimizer implementation used for optimizing pairs of nodes.
"""
import numpy as np
from numba import config, njit, prange, threading_layer

from smore.core.optimizer.helper.loss_function import compute_dot_product_loss


class PairOptimizer:  # pylint: disable=too-few-public-methods
    """
    Pair optimizer implementation used for optimizing pairs of nodes.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        total_update_times: int,
        lr: float = 0.025,
        l2_reg: float = 0.01,
    ):
        self.embeddings: np.ndarray = embeddings
        self.learning_rate: float = lr
        self.learning_rate_min: float = self.learning_rate * 0.0001
        self.λ: float = l2_reg  # pylint: disable=non-ascii-name, invalid-name
        self.loss: np.ndarray = np.zeros(self.embeddings.shape)
        self.total_update_times: int = total_update_times
        self.n_update: int = 0

    def _update_learning_rate(self, learning_rate: float):
        """
        Updates the learning rate internally.
        """
        if learning_rate < self.learning_rate_min:
            learning_rate = self.learning_rate_min
        self.learning_rate = learning_rate

    def dot_product_loss(
        self,
        training_edges: np.ndarray,
        l2_reg: bool = False,
    ):
        """
        Train the embeddings using the dot product loss by given training edges.
        """
        # TODO: Add progress bar to indicate the training progress.
        loss: np.ndarray = compute_dot_product_loss(self.embeddings, training_edges)
        if l2_reg:
            self.embeddings += self.learning_rate * (loss - self.λ * self.embeddings)
        else:
            self.embeddings += self.learning_rate * loss
        self.n_update += 1
        self._update_learning_rate(
            1.0
            - (
                self.n_update / self.total_update_times * 10 ** 6
            )  # TODO: make sample times multiplier configurable
        )
