#include<Eigen/SparseCholesky>
#include<pybind11/eigen.h>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<iostream>
#include<cstdint>
namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixXi;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::VectorX;

template <typename T>
void sparse_solve(void* out_ptr, void** data_ptr) {
	T* b_ptr = reinterpret_cast<T *>(data_ptr[0]);
  T* sparse_data_ptr = reinterpret_cast<T *>(data_ptr[1]);
  int* sparse_indices_ptr = reinterpret_cast<int *>(data_ptr[2]);    
  const int N_global = *reinterpret_cast<const int *>(data_ptr[3]);
  const std::int64_t V_size = *reinterpret_cast<const std::int64_t *>(data_ptr[4]);
  const bool forward = *reinterpret_cast<const bool *>(data_ptr[5]);
  VectorX<T> V_data = Map<const VectorX<T>>(sparse_data_ptr, V_size);
  MatrixXi V_indices = Map<const MatrixXi>(sparse_indices_ptr, V_size, 2);

  static int prev_N_global = 0;
  static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> forwardsolver;
  static Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> backwardsolver;

  if (N_global == prev_N_global) { 
  	// do nothing
  } else {
	  // create matrix, create solver, and analyze it
	  std::vector<Eigen::Triplet<T>> tripletList;
	  tripletList.reserve(V_size);
	  for (int i = 0; i < V_size; ++i) {
	    tripletList.push_back(Eigen::Triplet<T>(V_indices(i,0),V_indices(i,1),V_data(i)));
	  }
	  Eigen::SparseMatrix<T> V(N_global,N_global);
	  V.setFromTriplets(tripletList.begin(), tripletList.end());

	  prev_N_global = N_global;

	  forwardsolver.analyzePattern(V);
		forwardsolver.factorize(V);

		backwardsolver.analyzePattern(V.transpose());
		backwardsolver.factorize(V.transpose());
	}

  VectorX<T> b = Map<const VectorX<T>>(b_ptr,N_global);
  T* x_ptr = reinterpret_cast<T *>(out_ptr);
  if (forward) {
  	Map<VectorX<T>>(x_ptr, N_global) = forwardsolver.solve(b);
  } else {
  	Map<VectorX<T>>(x_ptr, N_global) = backwardsolver.solve(b);
  }
}



template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["sparse_solve_f32"] = EncapsulateFunction(sparse_solve<float>);
  dict["sparse_solve_f64"] = EncapsulateFunction(sparse_solve<double>);
  return dict;
}
PYBIND11_MODULE(custom_call_sparse_solve, m) { 
	m.def("registrations", &Registrations); 
}