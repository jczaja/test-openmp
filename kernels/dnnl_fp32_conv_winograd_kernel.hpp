#ifndef _DNNL_FP32_CONV_WINOGRAD_KERNEL
#define _DNNL_FP32_CONV_WINOGRAD_KERNEL

#include <kernels/dnnl_fp32_conv_kernel.hpp>

class DNNLConvWinogradKernel : public DNNLKernel<NumF, HeightF, WidthF>
{
  public:
    // Registration of kernel
    DNNLConvWinogradKernel();
    // Initialization
    // Params: dimensions
    virtual void Init(platform_info &pi, int n, int c, int h, int w);

    virtual void ShowInfo(bool cold_caches);

  private:
   void InitializeData(float* ptr, unsigned int sized);
};
#endif
