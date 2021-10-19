import pytest
from numpy import array, double

from smore.model.matrix_factorization import MatrixFactorization


@pytest.mark.parametrize(
    "edge_list",
    [
        array([[0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]),
        array(
            [
                [0, 1, 0.1],
                [0, 2, 0.1],
                [1, 2, 0.1],
                [1, 3, 0.1],
                [2, 3, 0.1],
                [3, 4, 0.1],
            ]
        ),
        array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4]]),
    ],
)
def test_loading_edge_list_type(edge_list):
    """
    Test matrix factorization working successfully.
    """
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.edge_list.dtype == double


@pytest.mark.parametrize(
    "edge_list, expected_node_num",
    [
        (
            array([[1, 2, 0.2], [1, 3, 0.1], [2, 4, 0.2], [3, 4, 0.3]]),
            4,
        ),
        (
            array([[1, 2, 0.2], [1, 3, 0.1], [2, 4, 0.2], [3, 4, 0.3], [5, 6, 0.5]]),
            6,
        ),
        (
            array([[1, 2], [1, 3], [2, 4], [3, 4], [5, 6], [7, 8]]),
            8,
        ),
    ],
)
def test_loading_node_number(edge_list, expected_node_num):
    """
    Test matrix factorization for loading edges if nodes number is equal
    """
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.node_num == expected_node_num


@pytest.mark.parametrize(
    "edge_list, expected_edge_num",
    [
        (
            array([[1, 2, 0.2], [1, 3, 0.1], [2, 4, 0.2], [3, 4, 0.3]]),
            4,
        ),
        (
            array([[1, 2, 0.2], [1, 3, 0.1], [2, 4, 0.2], [3, 4, 0.3], [5, 6, 0.5]]),
            5,
        ),
        (
            array([[1, 2], [1, 3], [2, 4], [3, 4], [5, 6], [7, 8]]),
            6,
        ),
    ],
)
def test_loading_edge_number(edge_list, expected_edge_num):
    """
    Test matrix factorization for loading edges if edges number is equal.
    """
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.edge_num == expected_edge_num


@pytest.mark.parametrize(
    "edge_list,expected_error",
    [
        (
            array(
                [
                    [1, 2, 3, 5],
                    [1, 2, 4, 4],
                    [1, 2, 3, 5],
                    [1, 2, 3, 3],
                ]
            ),
            ValueError,
        ),
        (
            array(
                [
                    [1],
                    [1],
                    [1],
                    [1],
                ]
            ),
            ValueError,
        ),
        (
            [
                [1, 2, 3],
                [1, 2, 4],
                [1, 2, 3],
                [1, 2, 3],
            ],
            ValueError,
        ),
    ],
)
def test_invalid_edge_list(edge_list, expected_error):
    """
    Test matrix factorization for loading edges
    """
    with pytest.raises(expected_error):
        MatrixFactorization(edge_list=edge_list)
