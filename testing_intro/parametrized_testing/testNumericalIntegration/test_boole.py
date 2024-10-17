import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x, 0, 1, 1, 1 / 2, 1e-10],
        [lambda x: x**2 * np.exp(-(x**3)), 0, 1, 2, (np.e - 1) / (3 * np.e), 1e-5],
    ],
)
def testIntegrationBooleParametrized(f, a, b, n, prediction, tol):
    result = integrationMethods.integrationBoole(f, a, b, n)
    assert np.abs(result - prediction) < tol
