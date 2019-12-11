#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/kernel.hpp>
#include<toolbox.h>

REGISTER_KERNEL(Kernel);

Kernel::Kernel() : buffer_(nullptr)
{
  // Register kernel
  kernels[std::string("sum")] = this;
}

void Kernel::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;
  sized_ = n*c*h*w;

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

void Kernel::ShowInfo(void)
{
    std::cout << std::endl << " Sum " << sized_<<"x1" <<  std::endl << std::endl;
}

inline void Kernel::RunSingle(void)
{
# ifdef GENERATE_ASSEMBLY
    asm volatile ("BEGIN sum Kernel");
# endif
  for(unsigned int i = 0; i< sized_; ++i) {
    result_ += buffer_[i];
  }
# ifdef GENERATE_ASSEMBLY
    asm volatile ("END sum Kernel");
# endif
}

void Kernel::Run(int num_reps)
{
#ifdef MEMORY_TRAFFIC_COUNT
    auto mt = ToolBox(true); // Just overwritting caches
    //mt.StartCounting();
#endif
#ifdef RUNTIME_TEST
    auto rt = Runtime(tsc_ghz_,false);
#endif
    for(int n = 0; n< num_reps; ++n) {
#ifdef RUNTIME_TEST
      rt.Start();
#endif
      RunSingle();  // Single iteration execution
#ifdef RUNTIME_TEST
      rt.Stop();
#endif
    }
#ifdef MEMORY_TRAFFIC_COUNT
    //mt.StopCounting();
#endif

}

