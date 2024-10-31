import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    # Error term: -1/90 *h**5 * f_4th_derivative(xi), where h=(b-a)/n and f(xi)=max(f(x)), a<x<b. 
    # The error term vanishes for polynomials up to 3rd order because of the 4th derivative. 
    # A very small numerical error term of 1e-10 is always added to the prediction to avoid floating point errors.
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x, 0, 1, 1, 1 / 2, 1e-10],
        [lambda x: x**2, 0, 1, 1, 1 / 3, 1e-10],
        [lambda x: x**3, 0, 1, 1, 1 / 4, 1e-10],
        [lambda x: x**2 * np.exp(-(x**3)), 0, 1, 5, (np.e - 1) / (3 * np.e), 1e-5],
    ],
)
def testIntegrationSimpsonParametrized(f, a, b, n, prediction, tol):
    result = integrationMethods.integrationSimpson(f, a, b, n)
    assert np.abs(result - prediction) < tol

# def testIntegrationSimpsonParametrized():
#     f = [lambda x:x, lambda x: x**2 * np.exp(-(x**3))]
#     a = 0
#     b = 1
#     n = [1,5]
#     prediction = [1/2, (np.e - 1) / (3 * np.e)]
#     tol = [1e-10, 1e-5]
#     for i in [0,1]:
#         result = integrationMethods.integrationSimpson(f[i], a, b, n[i])
#         assert np.abs(result - prediction[i]) < tol[i]