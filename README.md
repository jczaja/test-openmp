# test-openmp
Little testing program for various openmp concepts






article:
Counters do not show vcomis instructions which is crucial in search for maximal value eg. used in pooling and softmax


Operational intensity:
GFLOPS / DRAM bytes access

Measuring Memory traffic
LLC-loads for reading data
LLC-stores for writting data

perf -- from C++ , perf event for selected hardware event.

LLC to DRAM , prefetcher?

cmake math expr division seem to work in integer domain

Memory bound for small data comparing to cacheline


Perform kernel till it is at least one minute runtime.

Check execution of kernel of one time if execution time is less that one monute then
double repetitions. For extrenly small kernels requested num_repetitions may be bigger
that size of int. Small values are still evaluated further

How to compute runtime performance

To much memory movment if return value straing to cout

disable turbo boost for good runtime measures
enable turbo boost

LLC -> DRAM does not account for pre-fetcher

Cold measurement and warm measurement

benchmarking (two not dependant FMA faster tan one). why?
no chain dependency.

It seems throughput of FMA is 0.5
