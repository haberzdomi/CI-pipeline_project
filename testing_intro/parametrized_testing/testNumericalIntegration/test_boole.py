import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    # Error term: -8/945 *h**7 * f_6th_derivative(xi), where h=(b-a)/n and f(xi)=max(f(x)), a<x<b. 
    # The error term vanishes for polynomials up to 5th order because of the 6th derivative. 
    # A very small numerical error term of 1e-10 is always added to the prediction to avoid floating point errors.
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x, 0, 1, 1, 1 / 2, 1e-10],
        [lambda x: x**2, 0, 1, 1, 1 / 3, 1e-10],
        [lambda x: x**3, 0, 1, 1, 1 / 4, 1e-10],
        [lambda x: x**2 * np.exp(-(x**3)), 0, 1, 2, (np.e - 1) / (3 * np.e), 1e-5],
    ],
)
def testIntegrationBooleParametrized(f, a, b, n, prediction, tol):
    result = integrationMethods.integrationBoole(f, a, b, n)
    assert np.abs(result - prediction) < tol

# def testIntegrationBooleParametrized():
#     f = [lambda x:x, lambda x: x**2 * np.exp(-(x**3))]
#     a = 0
#     b = 1
#     n = [1,2]
#     prediction = [1/2, (np.e - 1) / (3 * np.e)]
#     tol = [1e-10, 1e-5]
#     for i in [0,1]:
#         result = integrationMethods.integrationBoole(f[i], a, b, n[i])
#         assert np.abs(result - prediction[i]) < tol[i]


## Comment for all tests:
## When we added the polynomials of second and third order, the rectangular and the trapezoidal integration were no 
## longer accurate. The new thresholds were chosen according to the error term calculation of the respective integration 
## methods (see tables in https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas). The simpson and boole methods 
## were exact for the third grade polynomial, since they only generate errors from the forth grade and higher.