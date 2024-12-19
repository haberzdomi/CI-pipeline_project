import timeit
from grid import grid
import numpy as np
from biotsavart import read_coils, read_currents, get_field_on_grid, get_field_on_grid_numba_parallel, calc_biotsavart, calc_biotsavart_vectorized

def get_runtimes(grid, coils, currents, grid_iterator):
    print("For loop:")
    print(
        timeit.timeit(
            stmt=lambda: grid_iterator(grid, coils, currents, calc_biotsavart),
            setup="from biotsavart import calc_biotsavart, "+grid_iterator.__name__+", coils; from grid import grid",
            number=1
        )
    )
    print("Vectorized:")
    print(
        timeit.timeit(
            stmt=lambda: grid_iterator(grid, coils, currents, calc_biotsavart_vectorized),
            setup="from biotsavart import calc_biotsavart_vectorized, "+grid_iterator.__name__+", coils; from grid import grid",
            number=1
        )
    )

test_grid=grid(20, 20, 20, 65, 267, 0, 2*np.pi, -154, 154)
coils = read_coils('coil_file')
currents = read_currents('current_file')

get_runtimes(test_grid, coils, currents, get_field_on_grid)
get_runtimes(test_grid, coils, currents, get_field_on_grid_numba_parallel)

## Vectorized version of calc_biotsavart is around 100 times faster