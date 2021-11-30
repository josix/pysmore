"""
Unit tests for the loss function.
"""
import numpy as np
import pytest

from pysmore.core.optimizer.helper.loss_function import dot_product


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
