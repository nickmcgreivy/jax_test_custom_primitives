#include<iostream>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<Eigen/SparseCholesky>
#include<pybind11/eigen.h>

using SparseM = Eigen::SparseMatrix<double>;
using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace py = pybind11;

class SparseSolver {
public:
	SparseSolver(const SparseM A) : A(A) {
		solver.analyzePattern(A);
		solver.factorize(A);
	}

	SparseSolver(const SparseSolver& s) {
		A = s.get_matrix();
		solver.analyzePattern(A);
		solver.factorize(A);
	}

	VectorXd solve(const VectorXd b) {
		return solver.solve(b);
	}

private:

	SparseM get_matrix() const {
		return A;
	}

	SparseM A;
	Eigen::SimplicialLDLT<SparseM> solver;
};


PYBIND11_MODULE(eigensolve, m) {
    py::class_<SparseSolver>(m, "SparseSolver")
    	.def(py::init<const SparseM>())
        .def("solve", &SparseSolver::solve);
}