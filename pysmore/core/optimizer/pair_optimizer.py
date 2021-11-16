"""
Pair optimizer implementation used for optimizing pairs of nodes.
"""
import numpy as np

from pysmore.core.optimizer.helper.loss_function import compute_raw_dot_product_loss


class PairOptimizer:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Pair optimizer implementation used for optimizing pairs of nodes.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        total_update_times: int,
        sample_size: int = 10 ** 6,
        lr: float = 0.025,
        l2_reg: float = 0.01,
    ):
        self.embeddings: np.ndarray = embeddings
        self.learning_rate: float = lr
        self.learning_rate_min: float = self.learning_rate * 0.0001
        self.λ: float = l2_reg  # pylint: disable=non-ascii-name, invalid-name
        self.total_update_times: int = total_update_times
        self.sample_size: int = sample_size
        self.n_update: int = 0
        self._loss: np.ndarray = np.zeros(self.embeddings.shape)

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
        self._loss = np.zeros(self.embeddings.shape)

    @property
    def loss(self):  # pragma: no cover
        """
        Returns the loss.
        """
        return self._loss.sum(dtype=np.float128)

    def compute_loss(
        self,
        training_edges: np.ndarray,
        l2_reg: bool = False,
    ):
        """
        Train the embeddings using the dot product loss by given training edges.
        """
        self._reset_loss()
        self._loss = compute_raw_dot_product_loss(self.embeddings, training_edges)
        if l2_reg:
            self.embeddings += self.learning_rate * (
                self._loss - self.λ * self.embeddings
            )
        else:
            self.embeddings += self.learning_rate * self._loss
        self.n_update += 1
        self._update_learning_rate(
            1.0 - (self.n_update / (self.total_update_times * self.sample_size))
        )
