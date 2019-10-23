#ifndef _BASEKERNEL
#define _BASEKERNEL

#include <string>

class BaseKernel
{
  public: 
    virtual void Init(platform_info &pi, int n, int c, int h, int w) = 0;
    virtual void Run(int num_reps) = 0;
};


#endif
