# jax_test_custom_primitives


This directory holds three custom primitives.

To run the scripts, clone the directory and type `make`.

`multiply_add` adds two floats or doubles.

`matmul` multiples two numpy arrays using an Eigen backend. `jit` of matmul does not work.

`spare_solve` can be ignored, for now.