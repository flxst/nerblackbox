import pytest
from typing import Optional


def pytest_approx(number: Optional[float]):
    """
    get acceptable pytest range for number
    --------------------------------------
    :param number: [float], e.g. 0.82
    :return: pytest range, e.g. 0.82 +- 0.01
    """
    if number is None:
        return None
    else:
        test_precision = 0.01
        return pytest.approx(number, abs=test_precision)
