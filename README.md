# test-openmp
Little testing program for various openmp concepts


# Using Intel openmp
pip install intel-openmp
pip show -f intel-openmp # Check where libiomp5.so is placed
cmake ../ -DINTEL_OMP_DIR=<directory where libiomp5.so is located>  <other options>




