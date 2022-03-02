import jax
from jax import lax, core, config, xla, jit, grad
from jax._src import api, abstract_arrays
from jax.interpreters import ad
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_solve
from scipy.linalg import lu_solve as np_lu_solve
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
from jaxlib import xla_client


sparse_solve_p = core.Primitive("sparse_solve")

def sparse_solve_prim(b, sparse_data, sparse_indices, N_global, forward=True):
    return sparse_solve_p.bind(b, sparse_data, sparse_indices, N_global, forward)

def sparse_solve_impl(b, sparse_data, sparse_indices, N_global, forward=True):
    raise Exception("Sparse solve prim shouldn't be called except from within JIT")

def sparse_solve_abstract_eval(b, sparse_data, sparse_indices, N_global, forward=True):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)

def sparse_solve_value_and_jvp(primals, tangents):
    (b, sparse_data, sparse_indices, N_global, forward) = primals
    (bt, _, _, _, _) = tangents
    primal_out = sparse_solve_prim(b, sparse_data, sparse_indices, N_global, forward=forward)
    output_tangent = sparse_solve_prim(bt, sparse_data, sparse_indices, N_global, forward=forward)
    return (primal_out, output_tangent)

def sparse_solve_transpose(ct, b, sparse_data, sparse_indices, N_global, forward=True):
    return (sparse_solve_prim(-ct, sparse_data, sparse_indices, N_global, forward=not forward), None, None, None, None)


import custom_call_sparse_solve

for _name, _value in custom_call_sparse_solve.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

def sparse_solve_xla_translation(c, bc, sparse_data, sparse_indices, N_global, forward):
    bc_shape = c.get_shape(bc)
    bc_dtype = bc_shape.element_type()
    bc_dims = bc_shape.dimensions()
    bc_shape = xla_client.Shape.array_shape(jnp.dtype(bc_dtype), bc_dims, (0,))
    out_shape = xla_client.Shape.array_shape(jnp.dtype(bc_dtype), bc_dims, (0,))
    data_shape = c.get_shape(sparse_data)
    data_dtype = data_shape.element_type()
    data_dims = data_shape.dimensions()
    data_shape = xla_client.Shape.array_shape(jnp.dtype(data_dtype), data_dims, (0,))
    indices_shape = c.get_shape(sparse_indices)
    indices_dtype = indices_shape.element_type()
    indices_dims = indices_shape.dimensions()
    indices_shape = xla_client.Shape.array_shape(jnp.dtype(indices_dtype), indices_dims, (0, 1))

    assert bc_dtype == data_dtype

    if bc_dtype == jnp.float32:
        op_name = b"sparse_solve_f32"
    elif bc_dtype == jnp.float64:
        op_name = b"sparse_solve_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {bc_dtype}")
    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(bc,
            sparse_data,
            sparse_indices,
            N_global,
            xla_client.ops.ConstantLiteral(c, data_dims[0]),
            forward,
            ),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(bc_shape,
            data_shape, 
            indices_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(bool), (), ()),
            ),
    )


xla.backend_specific_translations["cpu"][sparse_solve_p] = sparse_solve_xla_translation


sparse_solve_p.def_impl(sparse_solve_impl)
sparse_solve_p.def_abstract_eval(sparse_solve_abstract_eval)
ad.primitive_jvps[sparse_solve_p] = sparse_solve_value_and_jvp
ad.primitive_transposes[sparse_solve_p] = sparse_solve_transpose