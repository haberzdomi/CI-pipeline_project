import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x+x**2+x**3, 0, 1, 1, 13 / 12, 1e-10],
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



## When we added the polynomials of second and third order to the test function x: x the rectangular and the trapezoidal
## integration were no longer accurate. The new threshold were chosen according to the error functions of the
## respective integration methods. The simpson and boole methods were exact for the third grade polynomial, 
## since they only generate errors from the forth grade and higher.