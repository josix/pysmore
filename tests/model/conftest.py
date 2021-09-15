import pytest


@pytest.fixture()
def interaction_edge_list():
    return [
        (0, 1, 1),
        (0, 2, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 4, 1),
        (3, 5, 1),
        (4, 5, 1),
        (4, 6, 1),
    ]
