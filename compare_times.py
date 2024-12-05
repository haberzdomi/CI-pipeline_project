import timeit


print("For loop:")
print(
    timeit.timeit(
        "get_field_on_grid(read_grid('biotsavart.inp',1), read_coils('co_asd.dd'), read_currents('cur_asd.dd'), calc_biotsavart)",
        setup="from biotsavart import calc_biotsavart, calc_biotsavart_vectorized, read_coils, read_currents, read_grid, get_field_on_grid"

    )
)
print("Vectorized:")
print(
    timeit.timeit(
        "get_field_on_grid(read_grid('biotsavart.inp',1), read_coils('co_asd.dd'), read_currents('cur_asd.dd'), calc_biotsavart_vectorized)",
        setup="from biotsavart import calc_biotsavart, calc_biotsavart_vectorized, read_coils, read_currents, read_grid, get_field_on_grid"
    )
)