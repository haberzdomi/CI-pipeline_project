import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    # Error term: -1/12 *h**3 * f''(xi), where h=(b-a)/n and f(xi)=max(f(x)), a<x<b. 
    # Here h = (1-0)/1 = 1. For the linear term the second derivative is zero and the error term vanishes. 
    # A very small numerical error term of 1e-10 is always added to the prediction to avoid floating point errors.
    # The minus sign in the error term is because the trapezoidal method is a closed Newton-Cotes formula and therefore
    # the approximation always leads to an value smaller than the exact integral.
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x, 0, 1, 1, 1 / 2, 1e-10],
        [lambda x: x**2, 0, 1, 1, 1 / 3, 1/6+1e-10],
        [lambda x: x**3, 0, 1, 1, 1 / 4, 1/2+1e-10],
        [lambda x: x**2 * np.exp(-(x**3)), 0, 1, 80, (np.e - 1) / (3 * np.e), 1e-5],
    ],
)
def testIntegrationTrapezoidalParametrized(f, a, b, n, prediction, tol):
    result = integrationMethods.integrationTrapezoidal(f, a, b, n)
    assert np.abs(result - prediction) < tol

# def testIntegrationTrapezoidalParametrized():
#     f = [lambda x:x, lambda x: x**2 * np.exp(-(x**3))]
#     a = 0
#     b = 1
#     n = [1,80]
#     prediction = [1/2, (np.e - 1) / (3 * np.e)]
#     tol = [1e-10, 1e-5]
#     for i in [0,1]:
#         result = integrationMethods.integrationTrapezoidal(f[i], a, b, n[i])
#         assert np.abs(result - prediction[i]) < tol[i]
