def integrationRectangle(f, a, b, n):
    """
    Compute the 1D integral of the given function f with help of the rectangle (midpoint) rule.
    Yields an exact result for linear functions.

    Keyword arguments:
    f -- real function of on one variable
    a -- lower boundary of the integral
    b -- upper boundary of the integral
    n -- number of subdivisions

    Return value:
    Returns the value of the integral
    """

    delta_x = (b - a) / n
    value = 0.0
    for i in range(n):
        value += f(a + (i + 0.5) * delta_x)
    return delta_x * value


def integrationTrapezoidal(f, a, b, n):
    """
    Compute the 1D integral of the given function f with help of the trapezoidal rule.
    Yields an exact result for linear functions.

    Keyword arguments:
    f -- real function of on one variable
    a -- lower boundary of the integral
    b -- upper boundary of the integral
    n -- number of subdivisions

    Return value:
    Returns the value of the integral
    """

    delta_x = (b - a) / n
    value = 0.0
    for i in range(n):
        value += f(a + i * delta_x) + f(a + (i + 1) * delta_x)
    return delta_x / 2 * value


def integrationSimpson(f, a, b, n):
    """
    Compute the 1D integral of the given function f with help of Simpson's rule.
    Yields a numerically exact result for polynomials up to order ?.

    Keyword arguments:
    f -- real function of on one variable
    a -- lower boundary of the integral
    b -- upper boundary of the integral
    n -- number of subdivisions

    Return value:
    Returns the value of the integral
    """

    delta_x = (b - a) / n
    value = 0.0
    for i in range(n):
        value += (
            f(a + i * delta_x)
            + 4 * f(a + (i + 0.5) * delta_x)
            + f(a + (i + 1) * delta_x)
        )
    return delta_x / 6 * value


def integrationBoole(f, a, b, n):
    """
    Compute the 1D integral of the given function f with help of Boole's rule.
    Yields a numerically exact result for polynomials up to order ?.

    Keyword arguments:
    f -- real function of on one variable
    a -- lower boundary of the integral
    b -- upper boundary of the integral
    n -- number of subdivisions

    Return value:
    Returns the value of the integral
    """

    delta_x = (b - a) / n
    value = 0.0
    for i in range(n):
        value += (
            7 * f(a + i * delta_x)
            + 32 * f(a + (i + 0.25) * delta_x)
            + 12 * f(a + (i + 0.5) * delta_x)
            + 32 * f(a + (i + 0.75) * delta_x)
            + 7 * f(a + (i + 1) * delta_x)
        )
    return delta_x / 90 * value
