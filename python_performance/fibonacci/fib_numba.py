import numpy as np
import numba as nb


@nb.jit(nopython=True)
def fib_dynamic_nb_sequential(N):
    """
    Dynamic programming implementation of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    fib_seq[0] = 1
    fib_seq[1] = 1
    for i in nb.prange(2, N):
        fib_seq[i] = fib_seq[i - 1] + fib_seq[i - 2]
    return fib_seq


@nb.jit(nopython=True, parallel=True)
def fib_dynamic_nb_parallel(N):
    """
    Dynamic programming implementation of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    fib_seq[0] = 1
    fib_seq[1] = 1
    for i in nb.prange(2, N):
        fib_seq[i] = fib_seq[i - 1] + fib_seq[i - 2]
    return fib_seq


@nb.jit(nopython=True)
def fib_moivre_binet_nb_sequential(N):
    """
    Closed form solution of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    denominator = np.sqrt(5)

    for i in nb.prange(1, N + 1):
        fib_seq[i - 1] = (phi**i - psi**i) / denominator

    return fib_seq


@nb.jit(nopython=True, parallel=True)
def fib_moivre_binet_nb_parallel(N):
    """
    Closed form solution of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    denominator = np.sqrt(5)

    for i in nb.prange(1, N + 1):
        fib_seq[i - 1] = (phi**i - psi**i) / denominator

    return fib_seq
