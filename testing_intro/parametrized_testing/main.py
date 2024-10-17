import numpy as np
from numericalIntegration import integrationMethods


def run():
    f = lambda x: np.sqrt(1 - x**2)
    a = 0
    b = 1
    n = 10

    I_1 = integrationMethods.integrationRectangle(f, a, b, n)
    I_2 = integrationMethods.integrationTrapezoidal(f, a, b, n)
    I_3 = integrationMethods.integrationSimpson(f, a, b, n)
    I_4 = integrationMethods.integrationBoole(f, a, b, n)

    analytic_value = np.pi / 4

    E_1 = np.abs(analytic_value - I_1)
    E_2 = np.abs(analytic_value - I_2)
    E_3 = np.abs(analytic_value - I_3)
    E_4 = np.abs(analytic_value - I_4)

    print(
        "Analytic solution: {:{width}.{precision}f} ".format(
            analytic_value, width=12, precision=8
        )
    )
    print(
        "Rectangle value:   {:{width}.{precision}f}    Error: {:{width}.{precision}f}".format(
            I_1, E_1, width=12, precision=8
        )
    )
    print(
        "Trapezoidal value: {:{width}.{precision}f}    Error: {:{width}.{precision}f}".format(
            I_2, E_2, width=12, precision=8
        )
    )
    print(
        "Simpson value:     {:{width}.{precision}f}    Error: {:{width}.{precision}f}".format(
            I_3, E_3, width=12, precision=8
        )
    )
    print(
        "Boole value:       {:{width}.{precision}f}    Error: {:{width}.{precision}f}".format(
            I_4, E_4, width=12, precision=8
        )
    )


if __name__ == "__main__":
    run()
