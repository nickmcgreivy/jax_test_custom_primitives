import custom_call_matmul
import jax.numpy as jnp
from jaxlib import xla_client
from jax import abstract_arrays, core, xla
from jax import jit
from jax.config import config

# config.update("jax_enable_x64", True)






matmul_p = core.Primitive("matmul")  # Create the primitive

for _name, _value in custom_call_matmul.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)


def matmul_prim(x, y):
    return matmul_p.bind(x, y)


def matmul_impl(x, y):
    return x @ y


def matmul_abstract_eval(xs, ys):
    assert len(xs.shape) == 2
    assert len(ys.shape) == 2
    assert xs.shape[1] == ys.shape[0]
    return abstract_arrays.ShapedArray((xs.shape[0], ys.shape[1]), xs.dtype)


def matmul_xla_translation(c, xc, yc):
    xc_shape = c.get_shape(xc)
    yc_shape = c.get_shape(yc)
    dtype = xc_shape.element_type()
    assert yc_shape.element_type() == dtype

    xc_dims = xc_shape.dimensions()
    xc_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), xc_dims, (0, 1))

    yc_dims = yc_shape.dimensions()
    yc_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), yc_dims, (0, 1))
    out_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), (xc_dims[0], yc_dims[1]), (0, 1)
    )

    if dtype == jnp.float32:
        op_name = b"matmul_f32"
    elif dtype == jnp.float64:
        op_name = b"matmul_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(xc, yc, 
            xla_client.ops.ConstantLiteral(c, xc_dims[0]),
            xla_client.ops.ConstantLiteral(c, xc_dims[1]),
            xla_client.ops.ConstantLiteral(c, yc_dims[1])),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(xc_shape, 
            yc_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            ),
    )


matmul_p.def_impl(matmul_impl)
matmul_p.def_abstract_eval(matmul_abstract_eval)
xla.backend_specific_translations["cpu"][matmul_p] = matmul_xla_translation

x = jnp.ones((2, 3))
y = jnp.ones((3, 4))

res = matmul_prim(x, y)
jit_res = jit(matmul_prim)(x, y)
print("Result {} \nExpected {}".format(jit_res, res))