#include <iostream>
#include <pybind11/pybind11.h>


namespace py = pybind11;

// CPU //

template <typename T>
const void multiply_add(void* out_ptr, const void** data_ptr) {
    T x = ((T*) data_ptr[0])[0];
    T y = ((T*) data_ptr[1])[0];
    T z = ((T*) data_ptr[2])[0];
    T* out = (T*) out_ptr;
    out[0] = x*y + z;
}


template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["multiply_add_f32"] = EncapsulateFunction(multiply_add<float>);
  dict["multiply_add_f64"] = EncapsulateFunction(multiply_add<double>);
  return dict;
}

PYBIND11_MODULE(custom_call_multiply_add, m) { m.def("registrations", &Registrations); }