import pytest

from pysmore.core.optimizer import PairOptimizer


@pytest.fixture(scope="session")
def dummy_optimizer():
    """
    Dummy optimizer for testing.
    """
    node_num = 5
    dimension = 4
    return PairOptimizer(node_num, dimension, total_update_times=5)
