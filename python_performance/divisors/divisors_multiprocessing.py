import numpy as np
import multiprocessing as mp


def divisors_parallel_helper(N, i):
    return N % i == 0


def calculate_divisors_multiprocessing_async(number):
    max_divisor = number // 2

    pool = mp.Pool(mp.cpu_count())

    results = [
        pool.apply_async(divisors_parallel_helper, args=(number, i))
        for i in range(1, max_divisor + 1)
    ]
    bool_divisor_array = [False] + [p.get() for p in results]

    pool.close()
    pool.join()

    return bool_divisor_array


def calculate_divisors_multiprocessing_starmap(number):
    max_divisor = number // 2
    number_array = [number] * max_divisor

    pool = mp.Pool(mp.cpu_count())
    bool_divisor_array = pool.starmap(divisors_parallel_helper, zip(number_array, range(1, max_divisor + 1)))

    pool.close()
    pool.join()

    bool_divisor_array = [False] + bool_divisor_array

    return bool_divisor_array
