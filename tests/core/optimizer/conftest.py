import numpy as np
import pytest

from smore.core.optimizer import PairOptimizer


@pytest.fixture(scope="session")
def dummy_optimizer():
    """
    Dummy optimizer for testing.
    """
    node_num = 30
    dimension = 32
    return PairOptimizer(
        embeddings=np.random.rand(node_num, dimension),
        total_update_times=5,
    )
