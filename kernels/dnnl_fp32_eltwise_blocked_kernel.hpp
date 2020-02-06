#ifndef _DNNL_FP32_ELTWISE_BLOCKED_KERNEL
#define _DNNL_FP32_ELTWISE_BLOCKED_KERNEL

#include <kernels/dnnl_fp32_eltwise_kernel.hpp>

template<dnnl::algorithm algo>
class DNNLEltwiseBlockedKernel : public DNNLEltwiseKernel<algo, 0,0> 
{
  public:
    DNNLEltwiseBlockedKernel();
    // Initialization
    // Params: dimensions
    virtual void Init(platform_info &pi, int n, int c, int h, int w); 

    virtual void ShowInfo(bool cold_caches);

  private:
   void InitializeData(float* ptr, unsigned int sized);
};

#endif
