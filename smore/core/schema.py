"""
All types of data that will be used in SMORe
"""
from typing import Tuple, Union

Node = Union[int, str]
Edge = Tuple[Node, Node]
Weight = Union[int, float]
