import numpy as np


def calculate_divisors(number):
    max_divisor = number // 2
    bool_divisor_array = np.zeros(max_divisor + 1, dtype=np.bool_)

    for i in range(1, max_divisor + 1):
        if number % i == 0:
            bool_divisor_array[i] = True

    return bool_divisor_array

def calculate_divisors_vectorized(number):
    max_divisor = number // 2
    bool_divisor_array = np.zeros(max_divisor + 1, dtype=np.bool_)

    bool_divisor_array[1:] = number % np.arange(1, max_divisor + 1) == 0

    return bool_divisor_array
