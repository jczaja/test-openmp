#ifndef _DNNL_FP32_POOL_BLOCKED_KERNEL
#define _DNNL_FP32_POOL_BLOCKED_KERNEL

#include <kernels/dnnl_fp32_pool_kernel.hpp>

class DNNLPoolBlockedKernel : public DNNLPoolKernel<WidthF>
{

 public:
  // Registration of kernel
  DNNLPoolBlockedKernel();

  // Initialization
  // Params: dimensions
  virtual void Init(platform_info &pi, int n, int c, int h, int w);

  void ShowInfo(bool cold_caches);

  protected:
   void InitializeData(float* ptr, unsigned int sized);
};


#endif
