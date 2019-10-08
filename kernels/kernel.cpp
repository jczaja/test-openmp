#include <iostream>
#include <x86intrin.h>
#include<kernels/kernel.h>
#include<memory_traffic.h>

Kernel::Kernel(int n, int c, int h, int w) : sized_(n*c*h*w)
{
    int ret = posix_memalign((void**)&buffer_,64,n*c*h*w*sizeof(float));
    if (ret != 0) {
      std::cout << "Allocation error of bottom!" << std::endl;
      exit(-1);
    }
    // Init with some random data
    for(unsigned int i=0; i< (unsigned int )n*c*h*w; ++i) {
        buffer_[i] = i%13;
    }
}


Kernel::~Kernel()
{
    free(buffer_);
    buffer_ = nullptr;
}

unsigned long long Kernel::Run(int num_reps)
{
# ifdef GENERATE_ASSEMBLY
    asm volatile ("BEGIN Kernel");
# endif
    volatile auto result = 0.0f;
    auto start_t = __rdtsc();
#ifdef MEMORY_TRAFFIC_COUNT
    auto mt = MemoryTraffic();
    mt.StartCounting();
#endif
    for(int n = 0; n< num_reps; ++n) {
        for(unsigned int i = 0; i< sized_; ++i) {
          result  += buffer_[i];
        }
    }
#ifdef MEMORY_TRAFFIC_COUNT
    // Returning value to the cout stream directly makes lots of memory movement 
    auto ll = mt.StopCounting();
    //std::cout << "MemoryTraffic: " << mt.StopCounting() << std::endl;
    std::cout << "MemoryTraffic: " << ll << std::endl;
#endif
    auto measure = __rdtsc() - start_t;
    std::cout << "Sum result: " << result << std::endl;

# ifdef GENERATE_ASSEMBLY
    asm volatile ("END Kernel");
# endif
   return measure;
}

