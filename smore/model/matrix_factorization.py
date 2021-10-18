"""
Matrix Factorization Model
"""
from loguru import logger
from networkx import DiGraph
from numpy import double, ndarray
from tqdm import tqdm

from smore.model.base import BaseModel
from smore.pronet.sampler import EdgeSampler


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
        edge_list: ndarray,
        dimension: int = 64,
    ):  # pylint: disable=too-many-arguments
        super().__init__(dimension=dimension)
        self.edge_list = edge_list
        if not isinstance(self.edge_list, ndarray):
            raise ValueError("edge_list must be a numpy array")
        assert self.edge_list.shape[1] == 3, "edge_list must have 3 columns"
        self.edge_list = self.edge_list.astype(double)
        self.graph = DiGraph()
        self.edge_num = None
        self.node_num = None
        self.build_graph()
        self.sampler = EdgeSampler(graph=self.graph)

    def build_graph(self):
        """
        Build the graph from the given ``edge_list``
        """
        with tqdm(total=self.edge_list.shape[0], desc="Loading Graph") as progress_bar:
            for edge in self.edge_list:
                self.graph.add_edge(edge[0], edge[1], weight=edge[2])
                progress_bar.update(1)
        self.edge_num = self.graph.number_of_edges()
        self.node_num = self.graph.number_of_nodes()
        logger.info("#nodes: {}", self.node_num)
        logger.info("#edges: {}", self.edge_num)

    def train(self):
        """
        Train the model
        """


if __name__ == "__main__":

    def main():
        """
        Main function
        """
        import time

        from numpy import concatenate, ones, random

        edge_nums = [10 ** 7, 10 ** 6, 10 ** 5]
        for edge_num in edge_nums:
            logger.info("Running edge_num: {}".format(edge_num))
            edges = random.randint(0, 10000, size=(edge_num, 2))
            weights = ones((edge_num, 1))
            edge_list = concatenate((edges, weights), axis=1)
            loading_start = time.time()
            matrix_factorization = MatrixFactorization(edge_list=edge_list)
            loading_end = time.time()
            logger.info("Loading cost {:.2f}s", loading_end - loading_start)

    main()
