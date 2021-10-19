"""
Edge Sampler Implementation used for edge sampling from graph.
"""
from typing import List

from networkx import DiGraph
from numpy.random import Generator, default_rng

from smore.core.schema import Weight
from smore.core.utils import degree_distribution, in_degree_distribution


class EdgeSampler:  # pylint: disable=too-few-public-methods
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
        self.graph: DiGraph = graph
        self.n_nodes: int = graph.number_of_nodes()
        self.n_edges: int = graph.number_of_edges()
        self.rand_generator: Generator = default_rng()
        self.node_distribution: List[Weight] = degree_distribution(graph)
        self.negative_distribution: List[Weight] = in_degree_distribution(graph)
        self.edge_distribution: List[Weight] = [
            edge_with_data[2]
            for edge_with_data in self.graph.edges.data("weight", default=1)
        ]

    def sample_edge(self):
        """
        Sample edge from graph.

        :return: Edge sampled from graph.
        """
