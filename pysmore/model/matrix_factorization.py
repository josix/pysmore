"""
Matrix Factorization Model
"""
from typing import Optional

import numpy as np
from loguru import logger
from networkx import DiGraph
from tqdm import tqdm

from pysmore.core.optimizer import PairOptimizer
from pysmore.core.sampler import EdgeSampler
from pysmore.model.base import BaseModel


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
        sample_size: int = 1000000,
        lr: float = 0.025,
        l2_reg: float = 0.01,
    ):  # pylint: disable=too-many-arguments
        super().__init__(dimension=dimension)
        self.edge_list: np.ndarray = edge_list
        self.sample_times = sample_times
        self.sample_size = sample_size
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
        if self.node_num:
            self.optimizer: PairOptimizer = PairOptimizer(
                total_update_times=self.sample_times,
                sample_size=self.sample_size,
                node_num=self.node_num,
                dimension=self.dimension,
                lr=lr,
                l2_reg=l2_reg,
            )

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

    def train(self):  # pragma: no cover
        """
        Train the model
        """
        logger.info("Start training")
        for i in range(self.sample_times):
            edges = self.sampler.sample_edges(size=self.sample_size, with_weight=True)
            self.optimizer.update(edges, l2_reg=True)
            if i % 100 == 0:
                logger.info(
                    "Iteration {}/{} with loss {}",
                    i + 1,
                    self.sample_times,
                    self.optimizer.loss,
                )


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
            matrix_factorization = MatrixFactorization(
                edge_list=edge_list,
                sample_size=10000,
                sample_times=1000,
            )
            matrix_factorization.train()
            loading_end = time.time()
            logger.info("Training cost {:.2f}s", loading_end - loading_start)

    main()
