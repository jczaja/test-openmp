#ifndef _BASEKERNEL
#define _BASEKERNEL

#include <string>

class BaseKernel
{
  virtual void Init(void) = 0;
  virtual std::string name(void) = 0;
  virtual void Run(int num_reps) = 0;
};


#endif
