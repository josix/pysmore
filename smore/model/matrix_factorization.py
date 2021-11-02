"""
Matrix Factorization Model
"""
from typing import Optional

import numpy as np
from loguru import logger
from networkx import DiGraph
from tqdm import tqdm

from smore.core.optimizer import PairOptimizer
from smore.core.sampler import EdgeSampler
from smore.model.base import BaseModel


class MatrixFactorization(BaseModel):  # pylint: disable=too-many-instance-attributes
    """
    Matrix Factorization
    params:
        edge_list: Numpy 2D array of shape (n_edges, 3)
            where each row is an edge including source,
            target, and weight.
        dimension: The dimension of the generated embedding
    """

    def __init__(
        self,
        edge_list: np.ndarray,
        dimension: int = 64,
        sample_times: int = 5,
    ):  # pylint: disable=too-many-arguments
        super().__init__(dimension=dimension)
        self.edge_list: np.ndarray = edge_list
        self.sample_times = sample_times
        if not isinstance(self.edge_list, np.ndarray):
            raise ValueError("edge_list must be a numpy array")
        if self.edge_list.shape[1] > 3 or self.edge_list.shape[1] < 2:
            raise ValueError(
                "edge_list must be formatted in"
                " [source_node, target_node]/[sorce_node, target_node, weight]",
            )
        # TODO: Check node_id is int
        self.edge_list = self.edge_list.astype(
            np.double
        )  # Transform the type of weight to double
        self.graph: DiGraph = DiGraph()
        self.edge_num: Optional[int] = None
        self.node_num: Optional[int] = None
        self._build_graph()
        self.sampler: EdgeSampler = EdgeSampler(graph=self.graph)
        self.all_embedding: Optional[np.ndarray] = None
        self._initialize_embedding()
        if self.all_embedding is not None:
            self.optimizer: PairOptimizer = PairOptimizer(
                self.all_embedding, total_update_times=self.sample_times
            )

    def _initialize_embedding(self):
        """
        Initialize the embedding
        """
        self.all_embedding = np.random.rand(self.node_num, self.dimension)

    def _build_graph(self):
        """
        Build the graph from the given ``edge_list``
        """
        with tqdm(total=self.edge_list.shape[0], desc="Loading Graph") as progress_bar:
            for edge in self.edge_list:
                if len(edge) == 3:
                    self.graph.add_edge(
                        int(edge[0]),
                        int(edge[1]),
                        weight=edge[2],
                    )
                else:
                    self.graph.add_edge(
                        int(edge[0]),
                        int(edge[1]),
                        weight=1,
                    )
                progress_bar.update(1)
        self.edge_num = self.graph.number_of_edges()
        self.node_num = self.graph.number_of_nodes()
        logger.info("#nodes: {}", self.node_num)
        logger.info("#edges: {}", self.edge_num)

    def train(self):  # param: no cover
        """
        Train the model
        """
        if self.all_embedding is None:
            raise ValueError("Embedding is not initialized")
        with tqdm(
            total=self.sample_times, desc="Training"
        ) as progress_bar:  # TODO: Remove the progress bar and use the logger
            for i in range(self.sample_times):
                edges = self.sampler.sample_edges(
                    size=10 ** 6, with_weight=True
                )  # TODO: make sample times multiplier configurable
                self.optimizer.dot_product_loss(edges, l2_reg=True)
                progress_bar.update(1)


if __name__ == "__main__":  # pragma: no cover

    def main():  # pylint: disable=all
        """
        Main function
        """
        import time

        from numpy import concatenate, ones, random

        edge_nums = [10 ** 5, 5 * 10 ** 5, 10 ** 6, 3 * 10 ** 6, 5 * 10 ** 6]
        for edge_num in edge_nums:
            logger.info("Running edge_num: {}".format(edge_num))
            edges = random.randint(0, 10000, size=(edge_num, 2))
            weights = ones((edge_num, 1))
            edge_list = concatenate((edges, weights), axis=1)
            loading_start = time.time()
            matrix_factorization = MatrixFactorization(edge_list=edge_list)
            matrix_factorization.train()
            loading_end = time.time()
            logger.info("Loading cost {:.2f}s", loading_end - loading_start)

    main()
