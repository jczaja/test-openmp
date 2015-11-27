include /home/jczaja/TAU/tau-2.25/x86_64/lib/Makefile.tau-pdt-openmp-opari-trace 

TAU_OPTIONS = -optKeepFiles -optVerbose # keep intermediate files , show compilation process step by step

CC = $(TAU_COMPILER)$(TAU_OPTIONS) gcc -openmp
CXX = $(TAU_COMPILER)$(TAU_OPTIONS) g++ -openmp
FC = $(TAU_COMPILER)$(TAU_OPTIONS) gfortran -openmp 
LOADER = $(TAU_COMPILER)$(TAU_OPTIONS) gfortran -openmp

all: main.cpp.o
	$(CXX) -fopenmp -O3 -DNDEBUG   main.cpp.o  -o test-openmp -rdynamic 

main.cpp.o: /home/jczaja/test-openmp/main.cpp
	$(CXX) -fopenmp -O3 -DNDEBUG   -std=c++11 -march=native -Wall -o main.cpp.o -c /home/jczaja/test-openmp/main.cpp
clean:
	-rm test-openmp main.cpp.o *.o *.trc *.slog2 *.edf
show: all
	OMP_NUM_THREADS=4 ./test-openmp
	~/TAU/tau-2.25/x86_64/bin/tau_merge tautrace.0.0.* tautrace.trc    
	~/TAU/tau-2.25/x86_64/bin/tau2slog2 tautrace.trc tau.edf -o omp_trace.slog2 
	~/TAU/tau-2.25/x86_64/bin/jumpshot omp_trace.slog2  
