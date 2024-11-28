import numpy as np
import concurrent.futures as cf

def divisors_parallel_helper(N,i):
    return N % i == 0

def calculate_divisors_concurrent(number):
    max_divisor = number // 2

    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        results = [
            executor.submit(divisors_parallel_helper, number, i) for i in range(1, max_divisor + 1)
        ]
        bool_divisor_array = [False] + [p.result() for p in results]

    return bool_divisor_array
