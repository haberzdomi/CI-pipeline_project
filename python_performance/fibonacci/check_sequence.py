from fib_numba import fib_dynamic_numba, fib_moivre_binet_numba, fib_dynamic_numba_parallel,fib_moivre_binet_numba_parallel

print(fib_dynamic_numba(90))
print(fib_moivre_binet_numba(90))
print(fib_dynamic_numba_parallel(90))
print(fib_moivre_binet_numba_parallel(90))


## The dynamic parallel solution often cuts off at some point and displays only zeros from there. This is possibly
## the case because an accident occurs, when a recursive series is calculated with parallel processing.