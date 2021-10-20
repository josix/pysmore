"""
Unit tests for the sampler class
"""
import numpy as np
import pytest

from smore.core.sampler import EdgeSampler
from smore.core.utils import (
    degree_distribution,
    edge_distribution,
    in_degree_distribution,
    normalize,
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
    "array,expected_array",
    [
        (np.array([2, 4, 4]), [0.2, 0.4, 0.4]),
        ([2, 4, 4], [0.2, 0.4, 0.4]),
        ((2, 4, 4), [0.2, 0.4, 0.4]),
    ],
)
def test_normalize(array, expected_array):
    """
    Test normalize function
    """
    assert normalize(array) == pytest.approx(expected_array)


@pytest.mark.parametrize(
    "graph,expected_distribution,use_weight",
    [
        (mf_graph, [0.4, 0.2, 0.4, 0], True),
        (mf_graph, [2, 1, 1, 0], False),
    ],
)
def test_out_degree_distribution(graph, expected_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = out_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "graph,expected_distribution,use_weight",
    [
        (mf_graph, [0, 0.1, 0.3, 0.6], True),
        (mf_graph, [0, 1, 1, 2], False),
    ],
)
def test_in_degree_distribution(graph, expected_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = in_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "graph,expected_distribution,use_weight",
    [
        (mf_graph, [0.4, 0.3, 0.7, 0.6], True),
        (mf_graph, [2, 2, 2, 2], False),
    ],
)
def test_all_degree_distribution(graph, expected_distribution, use_weight):
    """
    Test degree_distribution working as expected.
    """
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "graph,expected_distribution,use_weight",
    [
        (mf_graph, [0.1, 0.3, 0.2, 0.4], True),
        (mf_graph, [1, 1, 1, 1], False),
    ],
)
def test_edge_distribution(graph, expected_distribution, use_weight):
    """
    Test edge_distribution working as expected.
    """
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = edge_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "graph,size",
    [
        (mf_graph, 5),
        (mf_graph, 0),
        (mf_graph, 100),
    ],
)
def test_sample_edge(graph, size):
    """
    Test sample_edge working as expected.
    """
    sampler = EdgeSampler(graph=graph)
    sampled_edge = sampler.sample_edges(size=size)
    assert sampled_edge.shape == (size, 2)
    for source_node, target_node in sampled_edge:
        assert (source_node, target_node) in graph.edges
        assert source_node in graph.nodes
        assert target_node in graph.nodes
