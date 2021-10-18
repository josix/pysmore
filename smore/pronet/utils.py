"""
Utilities for generating distribution and other SMORE helper functions.
"""
from typing import List

from networkx import DiGraph


def out_degree_distribution(graph: DiGraph, use_weight: bool = True) -> List:
    """
    Generate distribution from out degrees of the given graph.
    """
    if use_weight:
        return [graph.out_degree(node, weight="weight") for node in graph.nodes()]
    return [graph.out_degree(node) for node in graph.nodes()]


def in_degree_distribution(graph: DiGraph, use_weight: bool = True) -> List:
    """
    Generate distribution from in degrees of the given graph.
    """
    if use_weight:
        return [graph.in_degree(node, weight="weight") for node in graph.nodes()]
    return [graph.in_degree(node) for node in graph.nodes()]


def degree_distribution(graph: DiGraph, use_weight: bool = True) -> List:
    """
    Generate distribution from degrees of the given graph.
    """
    if use_weight:
        return [graph.degree(node, weight="weight") for node in graph.nodes()]
    return [graph.degree(node) for node in graph.nodes()]
