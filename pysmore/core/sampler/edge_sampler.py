"""
Edge Sampler Implementation used for edge sampling from graph.
"""

import numpy as np
from networkx import DiGraph
from numpy.random import Generator, default_rng

from pysmore.core.sampler.utils import (
    degree_distribution,
    edge_distribution,
    in_degree_distribution,
)


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
        self.node_distribution: np.ndarray = degree_distribution(graph)
        self.negative_distribution: np.ndarray = in_degree_distribution(graph)
        self.edge_distribution: np.ndarray = edge_distribution(graph)

    def sample_edges(self, size: int = 1000, with_weight=True) -> np.ndarray:
        """
        Sample edges from graph.

        :param size: Number of edges to sample.
        :return: Edge sampled from graph.
        """

        edges: np.ndarray
        if with_weight:
            edges = self.rand_generator.choice(
                [
                    [source_node, target_node, data["weight"]]
                    for source_node, target_node, data in self.graph.edges(data=True)
                    if "weight" in data
                ],
                size=size,
                p=self.edge_distribution,
            )
        else:
            edges = self.rand_generator.choice(
                list(self.graph.edges), size=size, p=self.edge_distribution
            )
        return edges
