
default:
	make multiply_add
	make matmul
	make sparsesolve

multiply_add:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` multiply_add_prim/custom_call_multiply_add.cpp -o multiply_add_prim/custom_call_multiply_add`python3-config --extension-suffix`
	python multiply_add_prim/multiply_add.py

matmul:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` matmul_prim/custom_call_matmul.cpp -o matmul_prim/custom_call_matmul`python3-config --extension-suffix`
	python matmul_prim/matmul.py

sparsesolve:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve_prim/sparse_solve.cpp -o sparse_solve_prim/custom_call_sparse_solve`python3-config --extension-suffix`
	python sparse_solve_prim/sparse_solve.py


clean: