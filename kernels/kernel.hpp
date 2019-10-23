#ifndef _MYKERNEL
#define _MYKERNEL

#include <string>
#include <toolbox.h>
#include <kernels/base_kernel.hpp>

class Kernel : public BaseKernel
{
 public:
  // Initialization
  // Params: dimensions
  Kernel(platform_info &pi, int n, int c, int h, int w);

  // Measured Execution of kernel
  // params: number of repetitions to execute 
  // returns: total time in cycles as measured by TSC
  void Run(int num_reps);

  // cleaning up and printing result
  ~Kernel() {
    std::cout << "Computed sum: " << result_ << std::endl;
    free(buffer_);
    buffer_ = nullptr;
  }
     
  std::string name() {
    return std::string("sum");
  }
 protected:
   void RunSingle(void);

 private:
   unsigned long long tsc_ghz_;
   unsigned int sized_;
   float *buffer_;
   float result_; 
};

#endif
