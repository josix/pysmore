import pytest
import numpy as np

from smore.model.matrix_factorization import MatrixFactorization
from smore.pronet.utils import (
    degree_distribution,
    in_degree_distribution,
    out_degree_distribution,
)


def get_mf_graph(edge_list: np.ndarray):
    """
    Sample graph from matrix factorization model
    """
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    return matrix_factorization.graph


def test_out_degree_distribution_with_weights():
    """
    Test out_degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = out_degree_distribution(graph=mf_graph, use_weight=True)
    assert distribution == pytest.approx([0.4, 0.2, 0.4, 0])


def test_out_degree_distribution_without_weights():
    """
    Test in_degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = out_degree_distribution(graph=mf_graph, use_weight=False)
    assert distribution == [2, 1, 1, 0]


def test_in_degree_distribution_with_weights():
    """
    Test out_degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = in_degree_distribution(graph=mf_graph, use_weight=True)
    assert distribution == pytest.approx([0, 0.1, 0.3, 0.6])


def test_in_degree_distribution_without_weights():
    """
    Test in_degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = in_degree_distribution(graph=mf_graph, use_weight=False)
    assert distribution == [0, 1, 1, 2]


def test_all_degree_distribution_with_weights():
    """
    Test degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = degree_distribution(graph=mf_graph, use_weight=True)
    assert distribution == pytest.approx([0.4, 0.3, 0.7, 0.6])


def test_all_degree_distribution_without_weights():
    """
    Test degree_distribution working as expected.
    """
    edge_list = np.array(
        [
            [1, 2, 0.1],
            [1, 3, 0.3],
            [2, 4, 0.2],
            [3, 4, 0.4],
        ]
    )
    mf_graph = get_mf_graph(edge_list)
    distribution = degree_distribution(graph=mf_graph, use_weight=False)
    assert distribution == [2, 2, 2, 2]
