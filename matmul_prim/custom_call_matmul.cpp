#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include<Eigen/Dense>
#include <iostream>
#include <cstdint>
namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixX;

template <typename T>
void matmul(void* out_ptr, void** data_ptr) {
    T* x_ptr = reinterpret_cast<T *>(data_ptr[0]);
    T* y_ptr = reinterpret_cast<T *>(data_ptr[1]);
    T* z_ptr = reinterpret_cast<T *>(out_ptr);
    const std::int64_t s1 = *reinterpret_cast<const std::int64_t *>(data_ptr[2]);
    const std::int64_t s2 = *reinterpret_cast<const std::int64_t *>(data_ptr[3]);
    const std::int64_t s3 = *reinterpret_cast<const std::int64_t *>(data_ptr[4]);
    MatrixX<T> x = Map<const MatrixX<T>>(x_ptr,s1,s2);
    MatrixX<T> y = Map<const MatrixX<T>>(y_ptr,s2,s3);

    Map<MatrixX<T>>(z_ptr, x.rows(), y.cols() ) = x * y;
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["matmul_f32"] = EncapsulateFunction(matmul<float>);
  dict["matmul_f64"] = EncapsulateFunction(matmul<double>);
  return dict;
}

PYBIND11_MODULE(custom_call_matmul, m) { 
  m.def("registrations", &Registrations); 
}