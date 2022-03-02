# jax_test_custom_primitives


This directory has three JAX custom primitives.

To run the scripts, clone the directory and type `make`.

`multiply_add` adds two floats or doubles, and does not use Eigen.

`matmul` multiples two numpy arrays using an Eigen backend. 

`spare_solve` is a JAX primitive operation which wraps an Eigen sparse matmul.
