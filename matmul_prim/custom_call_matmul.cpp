#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include<Eigen/Dense>
#include <iostream>
#include <cstdint>
namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::Matrix;


void matmul(void* out_ptr, void** data_ptr) {
    float* x_ptr = reinterpret_cast<float *>(data_ptr[0]);
    float* y_ptr = reinterpret_cast<float *>(data_ptr[1]);
    float* z_ptr = reinterpret_cast<float *>(out_ptr);
    const std::int64_t s1 = *reinterpret_cast<const std::int64_t *>(data_ptr[2]);
    const std::int64_t s2 = *reinterpret_cast<const std::int64_t *>(data_ptr[3]);
    const std::int64_t s3 = *reinterpret_cast<const std::int64_t *>(data_ptr[4]);
    MatrixXf x = Map<const MatrixXf>(x_ptr,s1,s2);
    MatrixXf y = Map<const MatrixXf>(y_ptr,s2,s3);
    Map<MatrixXf>(z_ptr, x.rows(), y.cols() ) = x * y;
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["matmul_f32"] = EncapsulateFunction(matmul);
  return dict;
}

PYBIND11_MODULE(custom_call_matmul, m) { 
  m.def("registrations", &Registrations); 
}