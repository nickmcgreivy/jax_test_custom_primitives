
default:
	make multiply_add
	make matmul


multiply_add:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` multiply_add_prim/custom_call_multiply_add.cpp -o multiply_add_prim/custom_call_multiply_add`python3-config --extension-suffix`
	python multiply_add_prim/multiply_add.py

matmul:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` matmul_prim/custom_call_matmul.cpp -o matmul_prim/custom_call_matmul`python3-config --extension-suffix`
	python matmul_prim/matmul.py


clean: