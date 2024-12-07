import timeit


if __name__ == "__main__":
    print("Dynamic programming solution:")
    print(
        timeit.timeit(
            "fib_dynamic(90)", setup="from fib_simple import fib_dynamic", number=100000
        )
    )

    print("Closed form solution:")
    print(
        timeit.timeit(
            "fib_moivre_binet(90)",
            setup="from fib_simple import fib_moivre_binet",
            number=100000,
        )
    )

    print("Dynamic programming solution - numba (sequential):")
    print(
        timeit.timeit(
            "fib_dynamic_nb_seq(90)",
            setup="from fib_numba import fib_dynamic_nb_seq",
            number=100000,
        )
    )

    print("Closed form solution - numba (sequential):")
    print(
        timeit.timeit(
            "fib_moivre_binet_nb_seq(90)",
            setup="from fib_numba import fib_moivre_binet_nb_seq",
            number=100000,
        )
    )

    print("Dynamic programming solution - numba (parallel):")
    print(
        timeit.timeit(
            "fib_dynamic_nb_parallel(90)",
            setup="from fib_numba import fib_dynamic_nb_parallel",
            number=100000,
        )
    )

    print("Closed form solution - numba (parallel):")
    print(
        timeit.timeit(
            "fib_moivre_binet_nb_parallel(90)",
            setup="from fib_numba import fib_moivre_binet_nb_parallel",
            number=100000,
        )
    )

## In the Python solutions the dynamic solution is faster than the closed form solution. The sequential closed form
## solution is the fastest. The parallel solutions are slower than the sequential solutions while the closed form
## is still faster than the dynamic solution.
