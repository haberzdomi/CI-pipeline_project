import numpy as np


def fib_dynamic(N):
    """
    Dynamic programming implementation of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    fib_seq[0] = 1
    fib_seq[1] = 1
    for i in range(2, N):
        fib_seq[i] = fib_seq[i - 1] + fib_seq[i - 2]
    return fib_seq


def fib_moivre_binet(N):
    """
    Closed form solution of the Fibonacci sequence
    up to the N-th Fibonacci number.
    """
    fib_seq = np.zeros(N)
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    denominator = np.sqrt(5)

    for i in range(1, N + 1):
        fib_seq[i - 1] = (phi**i - psi**i) / denominator

    return fib_seq
