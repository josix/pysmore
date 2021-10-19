import numpy as np
import pytest

from smore.core.utils import (
    degree_distribution,
    in_degree_distribution,
    out_degree_distribution,
)
from smore.model.matrix_factorization import MatrixFactorization


def get_mf_graph():
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


mf_graph = get_mf_graph()


@pytest.mark.parametrize(
    "graph,correct_distribution,use_weight",
    [
        (mf_graph, [0.4, 0.2, 0.4, 0], True),
        (mf_graph, [2, 1, 1, 0], False),
    ],
)
def test_out_degree_distribution(graph, correct_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """

    distribution = out_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(correct_distribution)


@pytest.mark.parametrize(
    "graph,correct_distribution,use_weight",
    [
        (mf_graph, [0, 0.1, 0.3, 0.6], True),
        (mf_graph, [0, 1, 1, 2], False),
    ],
)
def test_in_degree_distribution(graph, correct_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """
    distribution = in_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(correct_distribution)


@pytest.mark.parametrize(
    "graph,correct_distribution,use_weight",
    [
        (mf_graph, [0.4, 0.3, 0.7, 0.6], True),
        (mf_graph, [2, 2, 2, 2], False),
    ],
)
def test_all_degree_distribution(graph, correct_distribution, use_weight):
    """
    Test degree_distribution working as expected.
    """
    distribution = degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(correct_distribution)
