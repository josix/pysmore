"""
Unit tests for the sampler utilities
"""
import numpy as np
import pytest

from pysmore.core.sampler.utils import (
    degree_distribution,
    edge_distribution,
    in_degree_distribution,
    normalize,
    out_degree_distribution,
)


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
    "expected_distribution,use_weight",
    [
        ([0.4, 0.2, 0.4, 0], True),
        ([2, 1, 1, 0], False),
    ],
)
def test_out_degree_distribution(mf_graph, expected_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """
    graph = mf_graph
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = out_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "expected_distribution,use_weight",
    [
        ([0, 0.1, 0.3, 0.6], True),
        ([0, 1, 1, 2], False),
    ],
)
def test_in_degree_distribution(mf_graph, expected_distribution, use_weight):
    """
    Test out_degree_distribution working as expected.
    """
    graph = mf_graph
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = in_degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "expected_distribution,use_weight",
    [
        ([0.4, 0.3, 0.7, 0.6], True),
        ([2, 2, 2, 2], False),
    ],
)
def test_all_degree_distribution(mf_graph, expected_distribution, use_weight):
    """
    Test degree_distribution working as expected.
    """
    graph = mf_graph
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = degree_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)


@pytest.mark.parametrize(
    "expected_distribution,use_weight",
    [
        ([0.1, 0.3, 0.2, 0.4], True),
        ([1, 1, 1, 1], False),
    ],
)
def test_edge_distribution(mf_graph, expected_distribution, use_weight):
    """
    Test edge_distribution working as expected.
    """
    graph = mf_graph
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / np.sum(expected_distribution)
    distribution = edge_distribution(graph=graph, use_weight=use_weight)
    assert distribution == pytest.approx(expected_distribution)
