"""
Utilities for generating distribution and other SMORE helper functions.
"""
from typing import Union

import numpy as np
from networkx import DiGraph


def normalize(array: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Normalize the given array.
    """
    if isinstance(array, (list, tuple)):
        array = np.array(array)
    return array / np.sum(array)


def out_degree_distribution(graph: DiGraph, use_weight: bool = True) -> np.ndarray:
    """
    Generate distribution from out degrees of the given graph.
    """
    if use_weight:
        degree = [graph.out_degree(node, weight="weight") for node in graph.nodes()]
    else:
        degree = [graph.out_degree(node) for node in graph.nodes()]
    return normalize(np.array(degree))


def in_degree_distribution(graph: DiGraph, use_weight: bool = True) -> np.ndarray:
    """
    Generate distribution from in degrees of the given graph.
    """
    if use_weight:
        degree = [graph.in_degree(node, weight="weight") for node in graph.nodes()]
    else:
        degree = [graph.in_degree(node) for node in graph.nodes()]
    return normalize(np.array(degree))


def degree_distribution(graph: DiGraph, use_weight: bool = True) -> np.ndarray:
    """
    Generate distribution from degrees of the given graph.
    """
    if use_weight:
        degree = [graph.degree(node, weight="weight") for node in graph.nodes()]
    else:
        degree = [graph.degree(node) for node in graph.nodes()]
    return normalize(np.array(degree))


def edge_distribution(graph: DiGraph, use_weight: bool = True) -> np.ndarray:
    """
    Generate distribution from degrees of the given graph.
    """
    if not use_weight:
        return normalize(np.array([1 for _ in graph.edges.data("weight", default=1)]))
    return normalize(
        np.array(
            [
                edge_with_data[2]
                for edge_with_data in graph.edges.data("weight", default=1)
            ]
        )
    )
