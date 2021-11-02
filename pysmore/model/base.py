"""
Base model class
"""


class BaseModel:  # pylint: disable=too-few-public-methods # pragma: no cover
    """
    Base model class
    """

    def __init__(self, dimension: int = 64, sample_times: int = 5):
        """
        Initialize the model
        """
        self.dimension = dimension
        self.sample_times = sample_times

    def train(self):
        """
        Train the model
        """
        raise NotImplementedError
