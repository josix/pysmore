import pytest
from numpy import array, double

from smore.model.matrix_factorization import MatrixFactorization


def test_loading_edge_list():
    """
    Test matrix factorization working successfully.
    """
    edge_list = array(
        [
            [1, 2, 0.2],
            [1, 3, 0.1],
            [2, 4, 0.2],
            [3, 4, 0.3],
        ]
    )
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.edge_list.dtype == double


def test_loading_node_number():
    """
    Test matrix factorization for loading edges if nodes number is equal
    """
    edge_list = array(
        [
            [1, 2, 0.2],
            [2, 4, 0.2],
        ]
    )
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.node_num == 3


def test_loading_edge_number():
    """
    Test matrix factorization for loading edges if edges number is equal.
    """
    edge_list = array(
        [
            [1, 2, 0.2],
            [1, 3, 0.1],
            [2, 4, 0.2],
        ]
    )
    matrix_factorization = MatrixFactorization(edge_list=edge_list)
    assert matrix_factorization.edge_num == 3


def test_invalid_edge_list():
    """
    Test matrix factorization for loading edges
    """

    invalid_edge_list_with_error = [
        [
            array(
                [
                    [1, 2, 3, 5],
                    [1, 2, 4, 4],
                    [1, 2, 3, 5],
                    [1, 2, 3, 3],
                ]
            ),
            AssertionError,
        ],
        [
            [
                [1, 2, 3],
                [1, 2, 4],
                [1, 2, 3],
                [1, 2, 3],
            ],
            ValueError,
        ],
    ]
    for edge_list, error in invalid_edge_list_with_error:
        with pytest.raises(error):
            MatrixFactorization(edge_list=edge_list)
