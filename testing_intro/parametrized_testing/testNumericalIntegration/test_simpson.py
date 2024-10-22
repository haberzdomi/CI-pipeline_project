import numpy as np
import pytest
from numericalIntegration import integrationMethods


@pytest.mark.parametrize(
    "f,a,b,n,prediction,tol",
    [
        [lambda x: x+x**2+x**3, 0, 1, 1, 13 / 12, 1e-10],
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