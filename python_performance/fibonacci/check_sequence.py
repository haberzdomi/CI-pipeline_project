from fib_numba import (
    fib_dynamic_nb_seq,
    fib_moivre_binet_nb_seq,
    fib_dynamic_nb_parallel,
    fib_moivre_binet_nb_parallel,
)

print(fib_dynamic_nb_seq(90))
print(fib_moivre_binet_nb_seq(90))
print(fib_dynamic_nb_parallel(90))
print(fib_moivre_binet_nb_parallel(90))


# The dynamic parallel solution often leads to zero divisors at some point. Perhaps because in a recursive series the next element depends on the
# previous ones. Using parallel processing divisors might be calculated before the previous ones. The zero initialization then leads to zero results.
