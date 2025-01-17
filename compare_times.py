import timeit
from grid import grid
import numpy as np
from biotsavart import (
    read_coils,
    read_currents,
    get_field_on_grid,
    get_field_on_grid_numba_parallel,
    calc_biotsavart,
    calc_biotsavart_vectorized,
)


def get_runtimes(grid, coils, currents):
    time_seq_loop = timeit.timeit(
        stmt=lambda: get_field_on_grid(grid, coils, currents, calc_biotsavart),
        setup="from biotsavart import calc_biotsavart, "
        + get_field_on_grid.__name__
        + ", coils; from grid import grid",
        number=1,
    )
    print(f"Sequential loop time: {time_seq_loop} s")

    time_seq_vec = timeit.timeit(
        stmt=lambda: get_field_on_grid(
            grid, coils, currents, calc_biotsavart_vectorized
        ),
        setup="from biotsavart import calc_biotsavart_vectorized, "
        + get_field_on_grid.__name__
        + ", coils; from grid import grid",
        number=1,
    )
    print(f"Sequential vectorized time: {time_seq_vec} s")

    time_parallel_loop = timeit.timeit(
        stmt=lambda: get_field_on_grid_numba_parallel(
            grid, coils, currents, calc_biotsavart
        ),
        setup="from biotsavart import calc_biotsavart, "
        + get_field_on_grid_numba_parallel.__name__
        + ", coils; from grid import grid",
        number=1,
    )
    print(f"Parallel loop time: {time_parallel_loop} s")

    time_parallel_vec = timeit.timeit(
        stmt=lambda: get_field_on_grid_numba_parallel(
            grid, coils, currents, calc_biotsavart_vectorized
        ),
        setup="from biotsavart import calc_biotsavart_vectorized, "
        + get_field_on_grid_numba_parallel.__name__
        + ", coils; from grid import grid",
        number=1,
    )
    print(f"Parallel vectorized time: {time_parallel_vec} s")

    return time_seq_loop, time_seq_vec, time_parallel_loop, time_parallel_vec


test_grid = grid(12, 32, 24, 75, 267, 0, 2 * np.pi, -154, 154)
coils = read_coils("co_asd.dd")
currents = read_currents("cur_asd.dd")

times = get_runtimes(test_grid, coils, currents)

# For the above test_grid (large grid), the following times were measured:
# Sequential loop time:       79.1 s
# Sequential vectorized time: 11.6 s
# Parallel loop time:         19.6 s
# Parallel vectorized time:   10.1 s <-- fastest
