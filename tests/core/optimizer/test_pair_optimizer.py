"""
Unit tests for the pair optimizer
"""
import numpy as np
import pytest

from pysmore.core.optimizer import PairOptimizer
from pysmore.core.optimizer.helper.loss_function import compute_raw_dot_product_loss


@pytest.mark.parametrize(
    "learning_rate",
    [
        (0.025 * 0.0000000001),
        (0.025 * 0.0001),
        (0.025 * 0.001),
        (0.025 * 0.01),
    ],
)
def test_update_learning_rate(dummy_optimizer: PairOptimizer, learning_rate):
    """
    Test the dot product loss function
    """
    dummy_optimizer._update_learning_rate(  # pylint: disable=protected-access
        learning_rate
    )
    if dummy_optimizer.learning_rate_min > learning_rate:
        assert dummy_optimizer.learning_rate == pytest.approx(
            dummy_optimizer.learning_rate_min
        )
    else:
        assert dummy_optimizer.learning_rate == pytest.approx(learning_rate)


@pytest.mark.parametrize(
    "training_edges, l2_reg",
    [
        (
            np.array([[0, 1, 0], [1, 2, 1]]),
            False,
        ),
        (
            np.array([[0, 1, 0], [1, 2, 1]]),
            True,
        ),
        (
            np.array([[0, 1, 0.1], [1, 2, 1.2], [1, 2, 0.2]]),
            True,
        ),
        (
            np.array([[0, 1, 0.1], [1, 2, 1.2], [1, 2, 0.2]]),
            False,
        ),
    ],
)
def test_compute_loss(dummy_optimizer: PairOptimizer, training_edges, l2_reg):
    """
    Test the dot product loss function
    """
    embeddings = dummy_optimizer.embeddings.copy()
    update_times = dummy_optimizer.n_update
    raw_loss = compute_raw_dot_product_loss(dummy_optimizer.embeddings, training_edges)
    if l2_reg:
        embeddings += dummy_optimizer.learning_rate * (
            raw_loss - dummy_optimizer.λ * embeddings
        )
    else:
        embeddings += dummy_optimizer.learning_rate * raw_loss
    dummy_optimizer.compute_loss(training_edges, l2_reg)
    assert dummy_optimizer.embeddings == pytest.approx(embeddings)
    assert dummy_optimizer.n_update == update_times + 1
