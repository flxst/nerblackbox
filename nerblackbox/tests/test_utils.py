import pytest


def pytest_approx(number):
    """
    get acceptable pytest range for number
    --------------------------------------
    :param number: [float], e.g. 0.82
    :return: pytest range, e.g. 0.82 +- 0.01
    """
    test_precision = 0.01
    return pytest.approx(number, abs=test_precision)
