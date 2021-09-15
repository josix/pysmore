"""
Edge Sampler Implementation used for edge sampling from graph.
"""
from typing import List

from networkx import DiGraph


def out_degree_distribution(graph: DiGraph, use_weight: bool = True) -> List[int]:
    if use_weight:
        return [graph.out_degree(node, weight="weight") for node in graph.nodes()]
    return [graph.out_degree(node) for node in graph.nodes()]


def in_degree_distribution(graph: DiGraph, use_weight: bool = True) -> List[int]:
    if use_weight:
        return [graph.in_degree(node, weight="weight") for node in graph.nodes()]
    return [graph.in_degree(node) for node in graph.nodes()]


def degree_distribution(graph: DiGraph, use_weight: bool = True) -> List[int]:
    if use_weight:
        return [graph.degree(node, weight="weight") for node in graph.nodes()]
    return [graph.degree(node) for node in graph.nodes()]


class EdgeSampler:
    """
    Edge sampler usd for sampling training edges from graph.
    """

    def __init__(
        self,
        graph: DiGraph,
    ):
        """
        Initialize edge sampler.

        :param graph: Graph to sample edges from.
        """
        self.graph = graph
        self.node_distribution = degree_distribution(graph)
        self.negative_distribution = in_degree_distribution(graph)
        self.edge_distribution = [
            edge_with_data[2]  # edge_with_data = (source_node, target_node, weight)
            for edge_with_data in self.graph.edges.data("weight", default=1)
        ]
