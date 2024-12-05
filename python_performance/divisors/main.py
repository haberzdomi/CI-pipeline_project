import timeit

if __name__ == "__main__":
    m_div = 4000
    n_iterations = 200000
    print("Python solution:")
    print(
        timeit.timeit(
            "calculate_divisors("+str(m_div)+")",
            setup="from divisors_simple import calculate_divisors",
            number=n_iterations,
        )
    )

    print("NumPy (vectorized) solution:")
    print(
        timeit.timeit(
            "calculate_divisors_vectorized("+str(m_div)+")",
            setup="from divisors_simple import calculate_divisors_vectorized",
            number=n_iterations,
        )
    )

    # print("Concurrent solution:")
    # print(
    #     timeit.timeit(
    #         "calculate_divisors_concurrent(4000)",
    #         setup="from divisors_concurrent import calculate_divisors_concurrent",
    #         number=n_iterations,
    #     )
    # )

    # print("Multiprocessing async solution:")
    # print(
    #     timeit.timeit(
    #         "calculate_divisors_multiprocessing_async(4000)",
    #         setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_async",
    #         number=n_iterations,
    #     )
    # )

    # print("Multiprocessing starmap solution:")
    # print(
    #     timeit.timeit(
    #         "calculate_divisors_multiprocessing_starmap(4000)",
    #         setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_starmap",
    #         number=n_iterations,
    #     )
    #)

    # Effective with mid-low divisor input and high iteration number
    print("Numba solution:")
    print(
        timeit.timeit(
            "calculate_divisors_numba("+str(m_div)+")",
            setup="from divisors_numba import calculate_divisors_numba",
            number=n_iterations,
        )
    )
    # Effective with high divisor input
    print("Numba parallel solution:")
    print(
        timeit.timeit(
            "calculate_divisors_numba_parallel("+str(m_div)+")",
            setup="from divisors_numba import calculate_divisors_numba_parallel",
            number=n_iterations,
        )
    )


## 1) The Numpy (vectorized) solution finished the fastest.
## 2) The Python solution and the Numpy (vectorized) solution were roughly doubled. The times of the other methods 
## increased as well, but were much shorter than the doubled time, possibly due to the parallel processing 
## characteristic
## 3) After doubling the iteration number the times of all the tested methods were roughly doubled.
## 4) The three slowest approaches were commented out and only the Python and Numpy methods remained.
## 5) After adding the two numba methods the Numpy method was still the fastest.
## 6) 1. When there are many iterations. 2. No. 3. When the divisor input is relatively lower