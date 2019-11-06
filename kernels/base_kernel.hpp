#ifndef _BASEKERNEL
#define _BASEKERNEL

#include <string>

#define COMMA ,
#define REGISTER_KERNEL(T) extern std::unordered_map<std::string, BaseKernel*> kernels; \
                            static T objectT;

class BaseKernel
{
  public: 
    virtual void Init(platform_info &pi, int n, int c, int h, int w) = 0;
    virtual void Run(int num_reps) = 0;
    virtual void ShowInfo(void) = 0;
};


#endif
