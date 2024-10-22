import numpy as np
import pytest
from numericalIntegration import integrationMethods


# def testIntegrationRectangleLinear():
#     f = lambda x: x
#     a = 0
#     b = 1
#     n = 1

#     prediction = 1 / 2
#     tol = 1e-14
#     result = integrationMethods.integrationRectangle(f, a, b, n)

#     assert np.abs(result - prediction) < tol

@pytest.mark.parametrize(
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x+x**2+x**3, 0, 1, 1, 13 / 12, 8/3],
    ],
)
def testIntegrationRectangleLinear(f,a,b,n,prediction,tol):
    result = integrationMethods.integrationRectangle(f, a, b, n)
    assert np.abs(result - prediction) < tol