import numpy as np
import pytest

from smore.model.matrix_factorization import MatrixFactorization


@pytest.fixture(scope="session")
def mf_graph():
    """
    Sample graph from matrix factorization model
    """
    edge_list: np.ndarray = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    return matrix_factorization.graph
