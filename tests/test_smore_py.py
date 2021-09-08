"""
Test scripts for smore_py
"""
from smore import __version__


def test_version():
    """
    Test that the version is correct
    """
    assert __version__ == "0.1.0"
