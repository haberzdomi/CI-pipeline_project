import timeit

# 1.) n_iteration = 10, number (divisors) = 2000: NumPy (vectorized) solution is fastest and python solution is second fastest.

# 2.) n_iteration = 10, number (divisors) = 4000:
#   python solution - roughly doubled
#   NumPy (vectorized) solution - roughly 20% slower
#   Concurrent solution - roughly 20% slower
#   Multiprocessing async solution - roughly 20% slower
#   Multiprocessing starmap solution - roughly same
# Why might they not roughly double?
#   For normal python solution scales linear with number because
#   the for loop which calculates the divisors is 2 times larger.
#   For the other solutions, the divisors were calculated in parallel,
#   so one divisor calculation is not dependent on the others.
#   The length of the array is doubled, but not the time.


# 3.) n_iteration = 20, number (divisors) = 2000:
#   python solution - roughly doubled
#   NumPy (vectorized) solution - roughly doubled
#   Concurrent solution - roughly doubled
#   Multiprocessing async solution - roughly doubled
#   Multiprocessing starmap solution - roughly doubled

# 4.) Three slowest are: Concurrent, Multiprocessing async & Multiprocessing starmap

# 5.) n_iteration = 10, number (divisors) = 2000; numba solutions added: NumPy (vectorized) solution is fastest and python solution is second fastest.

# 6.) Play aroung with n_iteration and number (divisors):
# 6.1) Under which conditions does calculate_divisors_numba perform well?
#   calculate_divisors_numba performs best for high number of iterations and a moderate low input number for the divisor functions
#   e.g. n_iteration = 1000000, number (divisors) = 2000
# 6.2) Is calculate_divisors_numba_parallel always faster than calculate_divisors_numba?
#   No, if the input number for the divisor functions is high, parallel performs better.
#   e.g. n_iteration = 10000, number (divisors) = 1000000
# 6.3) When is it not?
#   If the input number for the divisor functions is low.
#   e.g. n_iteration = 100000, number (divisors) = 10
if __name__ == "__main__":
    n_iterations = 100000
    print("Python solution:")
    print(
        timeit.timeit(
            "calculate_divisors(10)",
            setup="from divisors_simple import calculate_divisors",
            number=n_iterations,
        )
    )

    print("NumPy (vectorized) solution:")
    print(
        timeit.timeit(
            "calculate_divisors_vectorized(10)",
            setup="from divisors_simple import calculate_divisors_vectorized",
            number=n_iterations,
        )
    )

    # print("Concurrent solution:")
    # print(
    #    timeit.timeit(
    #        "calculate_divisors_concurrent(10)",
    #        setup="from divisors_concurrent import calculate_divisors_concurrent",
    #        number=n_iterations,
    #    )
    # )

    # print("Multiprocessing async solution:")
    # print(
    #    timeit.timeit(
    #        "calculate_divisors_multiprocessing_async(10)",
    #        setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_async",
    #        number=n_iterations,
    #    )
    # )

    # print("Multiprocessing starmap solution:")
    # print(
    #    timeit.timeit(
    #        "calculate_divisors_multiprocessing_starmap(10)",
    #        setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_starmap",
    #        number=n_iterations,
    #    )
    # )

    print("Numba (sequential) solution:")
    print(
        timeit.timeit(
            "calculate_divisors_numba(10)",
            setup="from divisors_numba import calculate_divisors_numba",
            number=n_iterations,
        )
    )
    print("Numba (parallel) solution:")
    print(
        timeit.timeit(
            "calculate_divisors_numba_parallel(10)",
            setup="from divisors_numba import calculate_divisors_numba_parallel",
            number=n_iterations,
        )
    )
