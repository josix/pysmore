"""
Base model class
"""


class BaseModel:  # pylint: disable=too-few-public-methods # pragma: no cover
    """
    Base model class
    """

    def __init__(
        self, dimension: int = 64, sample_times: int = 5, sample_size: int = 1000000
    ):
        """
        Initialize the model
        """
        self.dimension = dimension
        self.sample_times = sample_times
        self.sample_size = sample_size

    def _build_graph(self):
        """
        Build the graph
        """
        raise NotImplementedError

    def _initialize_embedding(self):
        """
        Initialize the embedding
        """
        raise NotImplementedError

    def train(self):
        """
        Train the model
        """
        raise NotImplementedError
