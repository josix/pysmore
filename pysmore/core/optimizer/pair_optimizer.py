"""
Pair optimizer implementation used for optimizing pairs of nodes.
"""
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng

from pysmore.core.optimizer.helper.loss_function import compute_dot_product_update


@dataclass
class UpdateResult:  # pylint: disable=too-few-public-methods
    update_embedding: np.ndarray
    loss: float = 0.0


class PairOptimizer:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Pair optimizer implementation used for optimizing pairs of nodes.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        node_num: int,
        dimension: int,
        total_update_times: int,
        sample_size: int = 10 ** 6,
        lr: float = 0.025,
        l2_reg: float = 0.01,
    ):
        self.learning_rate: float = lr
        self.learning_rate_min: float = self.learning_rate * 0.0001
        self.λ: float = l2_reg  # pylint: disable=non-ascii-name, invalid-name
        self.total_update_times: int = total_update_times
        self.sample_size: int = sample_size
        self.n_update: int = 0
        self.loss: float = 0.0
        self.embeddings: np.ndarray = default_rng().uniform(
            low=-1, high=1, size=(node_num, dimension)
        )

    def _update_learning_rate(self, learning_rate: float):
        """
        Updates the learning rate internally.
        """
        if learning_rate < self.learning_rate_min:
            learning_rate = self.learning_rate_min
        self.learning_rate = learning_rate

    def _reset_loss(self):
        """
        Resets the loss.
        """
        self.loss = 0.0

    def update(
        self,
        training_edges: np.ndarray,
        l2_reg: bool = False,
    ):
        """
        Train the embeddings using the dot product loss by given training edges.
        """
        self._reset_loss()
        update_result = compute_dot_product_update(self.embeddings, training_edges)
        dot_product_update = UpdateResult(update_result[0], update_result[1])
        update_embedding: np.ndarray = dot_product_update.update_embedding
        self.loss = dot_product_update.loss
        if l2_reg:
            self.embeddings += self.learning_rate * (
                update_embedding - self.λ * self.embeddings
            )
        else:
            self.embeddings += self.learning_rate * update_embedding
        self.n_update += 1
        self._update_learning_rate(
            1.0 - (self.n_update / (self.total_update_times * self.sample_size))
        )
