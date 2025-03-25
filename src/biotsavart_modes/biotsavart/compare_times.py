import timeit
from biotsavart_modes.helpers.grid import GRID
import numpy as np
from biotsavart_modes.biotsavart.biotsavart import (
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
        setup="from biotsavart_modes.biotsavart.biotsavart import calc_biotsavart, "
        + get_field_on_grid.__name__
        + ", COILS; from biotsavart_modes.helpers.grid import GRID",
        number=5,
    )
    print(f"Sequential loop time: {time_seq_loop} s")

    time_seq_vec = timeit.timeit(
        stmt=lambda: get_field_on_grid(
            grid, coils, currents, calc_biotsavart_vectorized
        ),
        setup="from biotsavart_modes.biotsavart.biotsavart import calc_biotsavart_vectorized, "
        + get_field_on_grid.__name__
        + ", COILS; from biotsavart_modes.helpers.grid import GRID",
        number=5,
    )
    print(f"Sequential vectorized time: {time_seq_vec} s")

    time_parallel_loop = timeit.timeit(
        stmt=lambda: get_field_on_grid_numba_parallel(
            grid, coils, currents, calc_biotsavart
        ),
        setup="from biotsavart_modes.biotsavart.biotsavart import calc_biotsavart, "
        + get_field_on_grid_numba_parallel.__name__
        + ", COILS; from biotsavart_modes.helpers.grid import GRID",
        number=5,
    )
    print(f"Parallel loop time: {time_parallel_loop} s")

    time_parallel_vec = timeit.timeit(
        stmt=lambda: get_field_on_grid_numba_parallel(
            grid, coils, currents, calc_biotsavart_vectorized
        ),
        setup="from biotsavart_modes.biotsavart.biotsavart import calc_biotsavart_vectorized, "
        + get_field_on_grid_numba_parallel.__name__
        + ", COILS; from biotsavart_modes.helpers.grid import GRID",
        number=5,
    )
    print(f"Parallel vectorized time: {time_parallel_vec} s")

    return time_seq_loop, time_seq_vec, time_parallel_loop, time_parallel_vec


test_grid = GRID(24, 64, 48, 75.0, 267.0, 0, 2 * np.pi, -154.0, 150.4)
coils = read_coils("src/biotsavart_modes/input/co_asd.dd")
currents = read_currents("src/biotsavart_modes/input/cur_asd.dd")

times = get_runtimes(test_grid, coils, currents)

# For number=5 and GRID(24, 64, 48, 75.0, 267.0, 0, 2 * np.pi, -154.0, 150.4) the following times were measured:
# Sequential loop time: 2387.7 s -- devide by 5 --> 477.5 s per iteration
# Sequential vectorized time: 88.2 s -- devide by 5 --> 17.7 s per iteration
# Parallel loop time: 696.9 s -- devide by 5 --> 139.4 s per iteration
# Parallel vectorized time: 41.2 s -- devide by 5 --> 8.2 s per iteration <-- fastest
