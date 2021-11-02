from resource import classificator
import numpy as np
import pytest

@pytest.mark.parametrize("X, y, beta",  [(np.array([10, 14, 28, 19]), np.array([1]), np.zeros(7)),
                                        (np.array([10, 14, 28]), np.array([1]), np.zeros(5))])

def test_params_ValueError(X, y, beta):
    with pytest.raises(ValueError):
        classificator(X, y, beta)


@pytest.mark.parametrize("X, y, beta",  [(np.zeros(7), "1", np.zeros(7))])

def test_params_TypeError(X, y, beta):
    with pytest.raises(TypeError):
        classificator(X, y, beta)
