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

