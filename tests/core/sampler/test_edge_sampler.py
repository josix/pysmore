"""
Unit tests for the edge sampler
"""
import pytest

from pysmore.core.sampler.edge_sampler import EdgeSampler


@pytest.mark.parametrize(
    "size",
    [
        5,
        0,
        100,
    ],
)
def test_sample_edge(mf_graph, size):
    """
    Test sample_edge working as expected.
    """
    graph = mf_graph
    sampler = EdgeSampler(graph=graph)
    sampled_edge_with_weight = sampler.sample_edges(size=size, with_weight=True)
    sampled_edge_without_weight = sampler.sample_edges(size=size, with_weight=False)
    assert sampled_edge_with_weight.shape == (size, 3)
    assert sampled_edge_without_weight.shape == (size, 2)
    for source_node, target_node, *_ in [
        *sampled_edge_with_weight,
        *sampled_edge_without_weight,
    ]:
        assert (source_node, target_node) in graph.edges
        assert source_node in graph.nodes
        assert target_node in graph.nodes
