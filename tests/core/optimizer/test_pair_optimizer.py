"""
Unit tests for the pair optimizer
"""
import pytest

from smore.core.optimizer import PairOptimizer


@pytest.mark.skip()
@pytest.mark.parametrize(
    "learning_rate_min,learning_rate",
    [
        (0.025 * 0.0001, 0.025),
        (0.025 * 0.001, 0.025),
        (0.025 * 0.01, 0.025),
    ],
)
def test_update_learning_rate(
    dummy_optimizer: PairOptimizer, learning_rate_min, learning_rate
):
    """
    Test the dot product loss function
    """
    pass
