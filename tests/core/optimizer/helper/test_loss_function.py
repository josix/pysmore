"""
Unit tests for the loss function.
"""
import numpy as np
import pytest

from pysmore.core.optimizer.helper.loss_function import (
    compute_dot_product_update,
    dot_product,
)


@pytest.mark.parametrize(
    "vector1,vector2, expected_output",
    [
        (np.array([1, 2]), np.array([1, 2]), 5),
        (np.array([0.1, 0.3, 0.2]), np.array([0.2, 0.4, 0.4]), 0.22),
    ],
)
def test_dot_product(vector1, vector2, expected_output):
    """
    Test the dot product of two vectors.
    """
    dot_product_output = dot_product(vector1, vector2)
    assert pytest.approx(dot_product_output) == expected_output


@pytest.mark.parametrize(
    "embeddings, training_edges, expected_loss",
    [
        (
            np.array([[1, 1], [1, 0], [1, 2]]),
            np.array([[0, 1, 0], [1, 2, 1]]),
            np.array([[-1, -1], [-1, 0], [0, 0]]),
        )
    ],
)
def test_compute_dot_product_update(embeddings, training_edges, expected_loss):
    """
    Test dot product loss given embeddings and training edges.
    """
    update_embedding = compute_dot_product_update(embeddings, training_edges)[0]
    assert update_embedding == pytest.approx(expected_loss)
