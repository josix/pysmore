"""
Base model class
"""
from pathlib import Path


class BaseModel:
    """
    Base model class
    """

    def __init__(self, dimension: int = 64):
        """
        Initialize the model
        """
        self.dimension = dimension

    def train(self):
        """
        Train the model
        """
        raise NotImplementedError

    def export(self, path: Path):
        """
        Export the model
        """
        raise NotImplementedError
