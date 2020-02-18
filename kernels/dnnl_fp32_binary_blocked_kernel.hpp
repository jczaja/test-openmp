#ifndef _DNNL_FP32_BINARY_BLOCKED_KERNEL
#define _DNNL_FP32_BINARY_BLOCKED_KERNEL

#include <kernels/dnnl_fp32_binary_kernel.hpp>

template<dnnl::algorithm algo>
class DNNLBinaryBlockedKernel : public DNNLBinaryKernel<algo> 
{
  public:
    DNNLBinaryBlockedKernel();
    // Initialization
    // Params: dimensions
    virtual void Init(platform_info &pi, int n, int c, int h, int w); 

    virtual void ShowInfo(bool cold_caches);

    virtual ~DNNLBinaryBlockedKernel();

  private:
   void InitializeData(float* ptr, unsigned int sized);
};


#endif
