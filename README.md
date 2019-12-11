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

Memory object where allocated on heap to foul memory prefetchet


Memory throughput test is done without fixing CPU clocks


One thread cannot fully use memory throughput


When GOMP_CPU_AFFINITY set then then all in one thread then memory throughput is very 
low as multithreading has to work on one core

When not considering Prefetchers and non-temporaral instructions traffic computed is half of theoretical estimation which is minimum


Memory measurement varies due ot operating system work like pagin in some pages 

GEMM DNNL conv has software prefetcher instructions


Runtime placing:
Work - FLOPS to perf algorithm
Runtime - Time[cycles] of execution of single instance of algorithm

TRAFFIC:
a) LLC MISS:
83736064
b) LLC STORE:
33792576
c) LLC-PREFETCHES
0
d)LLC-PREFETCH-MISSES
0
 



4*(100*227*227*3 + 96*3*11*11 + 100*96*55*55) = 178134192/1024/1024 = 169 MB 
4*(200*227*227*3 + 96*3*11*11 + 200*96*55*55) = 356128992/1024/1024 = 339 MB 
                                                83736064  # Cache miss
READ:   61974192
WRITE: 116160000

Operating system is included
0 reps : 1300 MB
1 reps : 2600 MB
On operating lots of services result is not


Counters are bad as:
runtime: 0.32 s
memory traffic: 18747813120 (18 GB)




operation with sleep:
1222592

BAR adress:
sudo setpci -s 0:0.0 0x48.l

SKX laptop:
BAR - 0xfed10001


testing on sum algorithm

Disabling prefetcher(msr-tools):
wrmsr -p0 0x1a4 1
rdmsr -p0 0x1a4

HSW (BRIX):
47307712 (With prefetch)
55331392 (without prefetch)
actually no diffrence


sum (N=10000):
40128  # NO Memory prefetch 
10432  # Memory prefetcher


Problem is that we have half of memory transfer computed from cache misses
And more than possible memory transfer when DRAM_READ is conidered
it seems that software memory prefetcher does not change statr of processor so no cache miss

Due to security reasons openning /dev/mem is not available. Reading perf even via BAR + data_read adress gives very
big numbers much higher than capacity of bandwith.

Most promising is perf stat -e data_reads <program>
It is system wide measure and require root priviligies to be made


Patch to have PMU available:
https://lwn.net/Articles/585372/

Experiments with Relu showed that PMU of core are not good as relu is implemented as max(x,0) implemented
via vcomp are not counted in PMU events. Hence work is undercomputed


Layer norm for inference is memory bound as we read input, mean and variance and do not perform much
actual arithmetic operations. It may be good to make a forward training for comparison 

