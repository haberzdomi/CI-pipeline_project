import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    # Error term: 1/3 *h**3 * f''(xi), where h=(b-a)/(n+2) and f(xi)=max(f(x)), a<x<b. 
    # Here h = (1-0)/2 = 1/2. For the linear term the second derivative is zero and the error term vanishes. 
    # A very small numerical error term of 1e-10 is always added to the prediction to avoid floating point errors.
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x, 0, 1, 1, 1 / 2, 1e-10],
        [lambda x: x**2, 0, 1, 1, 1 / 3, 1/12+1e-10],
        [lambda x: x**3, 0, 1, 1, 1/4, 1/4+1e-10],
    ],
)
def testIntegrationRectangleLinear(f,a,b,n,prediction,tol):
    result = integrationMethods.integrationRectangle(f, a, b, n)
    assert np.abs(result - prediction) < tol

# def testIntegrationRectangleLinear():
#     f = lambda x: x
#     a = 0
#     b = 1
#     n = 1
#
#     prediction = 1 / 2
#     tol = 1e-14
#     result = integrationMethods.integrationRectangle(f, a, b, n)
#
#     assert np.abs(result - prediction) < tol
