import timeit

if __name__ == "__main__":
    n_iterations = 10
    print("Python solution:")
    print(
        timeit.timeit(
            "calculate_divisors(2000)",
            setup="from divisors_simple import calculate_divisors",
            number=n_iterations,
        )
    )

    print("NumPy (vectorized) solution:")
    print(
        timeit.timeit(
            "calculate_divisors_vectorized(2000)",
            setup="from divisors_simple import calculate_divisors_vectorized",
            number=n_iterations,
        )
    )

    print("Concurrent solution:")
    print(
        timeit.timeit(
            "calculate_divisors_concurrent(2000)",
            setup="from divisors_concurrent import calculate_divisors_concurrent",
            number=n_iterations,
        )
    )

    print("Multiprocessing async solution:")
    print(
        timeit.timeit(
            "calculate_divisors_multiprocessing_async(2000)",
            setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_async",
            number=n_iterations,
        )
    )

    print("Multiprocessing starmap solution:")
    print(
        timeit.timeit(
            "calculate_divisors_multiprocessing_starmap(2000)",
            setup="from divisors_multiprocessing import calculate_divisors_multiprocessing_starmap",
            number=n_iterations,
        )
    )
