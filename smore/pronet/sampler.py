"""
Edge Sampler Implementation used for edge sampling from graph.
"""

from networkx import DiGraph

from smore.pronet.utils import degree_distribution, in_degree_distribution


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

    def sample_edge(self):
        """
        Sample edge from graph.

        :return: Edge sampled from graph.
        """
        pass
