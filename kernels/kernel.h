#ifndef _MYKERNEL
#define _MYKERNEL

#include <string>
class Kernel
{
 public:
  // Initialization
  // Params: dimensions
  Kernel(int n, int c, int h, int w);

  // Measured Execution of kernel
  // params: number of repetitions to execute 
  // returns: total time in cycles as measured by TSC
  unsigned long long Run(int num_reps);

  // cleaning up
  ~Kernel();
     
  std::string name() {
    return std::string("Sequence Sum");
  }

 private:
   unsigned int sized_;
   float *buffer_;
};

#endif
