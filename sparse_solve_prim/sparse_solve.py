import jax
from jax import lax, core, config
from jax._src import api, abstract_arrays
from jax.interpreters import ad
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_solve
from scipy.linalg import lu_solve as np_lu_solve
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

import eigensolve

sparse_solve_p = core.Primitive("sparse_solve")


def sparse_solve_prim(b, solver=None, solver_T=None):
    return sparse_solve_p.bind(b, solver=solver, solver_T=solver_T)


def sparse_solve_impl(b, solver=None, solver_T=None):
    return jnp.asarray(solver.solve(np.asarray(b)))


sparse_solve_p.def_impl(sparse_solve_impl)


def sparse_solve_abstract_eval(b, solver=None, solver_T=None):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


sparse_solve_p.def_abstract_eval(sparse_solve_abstract_eval)


def sparse_solve_value_and_jvp(primals, tangents, solver=None, solver_T=None):
    (b,) = primals
    (bt,) = tangents
    primal_out = sparse_solve_prim(b, solver=solver, solver_T=solver_T)
    output_tangent = sparse_solve_prim(bt, solver=solver, solver_T=solver_T)
    return (primal_out, output_tangent)


ad.primitive_jvps[sparse_solve_p] = sparse_solve_value_and_jvp


def sparse_solve_transpose(ct, b, solver=None, solver_T=None):
    return (sparse_solve_prim(-ct, solver=solver_T, solver_T=solver),)


ad.primitive_transposes[sparse_solve_p] = sparse_solve_transpose


V = jnp.asarray([[2.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
b = jnp.asarray([1.0, 2.0, 3.0])
bt = jnp.asarray([4.0, -1.0, 1.0])

solver = eigensolve.SparseSolver(csr_matrix(V))
solver_T = eigensolve.SparseSolver(csr_matrix(V.T))


f = lambda b: jnp.sum(sparse_solve_prim(b, solver=solver, solver_T=solver_T) ** 2)
f_grad = api.grad(f)
print((sparse_solve_prim(b, solver=solver, solver_T=solver_T)))
print(api.jvp(lambda b: sparse_solve_prim(b, solver=solver, solver_T=solver_T), (b,), (bt,)))
print(f_grad(b))

"""
# GOAL IS TO GET THIS WORKING!
print(f_jit(b))
"""