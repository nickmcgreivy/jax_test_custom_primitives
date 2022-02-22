import custom_call_multiply_add
import jax.numpy as jnp
from jaxlib import xla_client
from jax import abstract_arrays, core, xla
from jax import jit
import numpy as np
from jax.config import config

# config.update("jax_enable_x64", True)


multiply_add_p = core.Primitive("multiply_add")  # Create the primitive

for _name, _value in custom_call_multiply_add.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)


def multiply_add_prim(x, y, z):
    return multiply_add_p.bind(x, y, z)


def multiply_add_impl(x, y, z):
    return np.add(np.multiply(x, y), z)


def multiply_add_abstract_eval(xs, ys, zs):
    assert xs.shape == ys.shape
    assert xs.shape == zs.shape
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


def multiply_add_xla_translation(c, xc, yc, zc):
    xc_shape = c.get_shape(xc)
    yc_shape = c.get_shape(yc)
    zc_shape = c.get_shape(zc)
    dtype = xc_shape.element_type()
    assert yc_shape.element_type() == dtype
    assert zc_shape.element_type() == dtype
    if dtype == np.float32:
        op_name = b"multiply_add_f32"
    elif dtype == np.float64:
        op_name = b"multiply_add_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(xc, yc, zc),
        shape_with_layout=xla_client.Shape.array_shape(dtype, (), ()),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(dtype, (), ()),
            xla_client.Shape.array_shape(dtype, (), ()),
            xla_client.Shape.array_shape(dtype, (), ()),
        ),
    )


multiply_add_p.def_impl(multiply_add_impl)
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)
xla.backend_specific_translations["cpu"][multiply_add_p] = multiply_add_xla_translation

x, y, z = (1.0, 2.0, 3.0)

res = multiply_add_prim(x, y, z)
jit_res = jit(multiply_add_prim)(x, y, z)

print("Result {} Expected {}".format(jit_res, res))