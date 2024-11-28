import numpy as np
import numba as nb

@nb.jit(nopython=True)
def calculate_divisors_numba(number):
    max_divisor = number // 2
    bool_divisor_array = np.zeros(max_divisor + 1, dtype=np.bool_)

    for i in range(1, max_divisor + 1):
        if number % i == 0:
            bool_divisor_array[i] = True

    return bool_divisor_array

@nb.jit(nopython=True, parallel=True)
def calculate_divisors_numba_parallel(number):
    max_divisor = number // 2
    bool_divisor_array = np.zeros(max_divisor + 1, dtype=np.bool_)

    for i in nb.prange(1, max_divisor + 1):
        if number % i == 0:
            bool_divisor_array[i] = True

    return bool_divisor_array
