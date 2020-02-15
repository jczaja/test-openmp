#ifndef _DNNL_FP32_ELTWISE_BLOCKED_KERNEL
#define _DNNL_FP32_ELTWISE_BLOCKED_KERNEL

#include <kernels/dnnl_fp32_eltwise_kernel.hpp>

template<dnnl::algorithm algo, int alpha, int beta>
class DNNLEltwiseBlockedKernel : public DNNLEltwiseKernel<algo, alpha, beta> 
{
  public:
    DNNLEltwiseBlockedKernel();
    // Initialization
    // Params: dimensions
    virtual void Init(platform_info &pi, int n, int c, int h, int w); 

    virtual void ShowInfo(bool cold_caches);

    virtual ~DNNLEltwiseBlockedKernel();

  private:
   void InitializeData(float* ptr, unsigned int sized);
};

#endif
