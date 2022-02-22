#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include<Eigen/Dense>
#include <iostream>
namespace py = pybind11;

// CPU //

template <typename T>
const void matmul(void* out_ptr, const void** data_ptr) {
    T x = ((T*) data_ptr[0])[0];
    T y = ((T*) data_ptr[1])[0];

    T* out = (T*) out_ptr;
    out[0] = x * y;
}


template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["matmul_f64"] = EncapsulateFunction(matmul<Eigen::MatrixXd>);
  dict["matmul_f32"] = EncapsulateFunction(matmul<Eigen::MatrixXf>);
  return dict;
}

PYBIND11_MODULE(custom_call_matmul, m) { m.def("registrations", &Registrations); }