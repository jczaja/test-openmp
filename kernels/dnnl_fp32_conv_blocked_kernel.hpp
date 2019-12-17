#ifndef _DNNL_FP32_BLOCKED_KERNEL
#define _DNNL_FP32_BLOCKED_KERNEL

#include <kernels/dnnl_fp32_conv_kernel.hpp>

class DNNLConvBlockedKernel : public DNNLKernel<NumF, HeightF, WidthF>
{
  public:
    // Registration of kernel
    DNNLConvBlockedKernel();
    // Initialization
    // Params: dimensions
    void Init(platform_info &pi, int n, int c, int h, int w);

    // Registration of kernel
    ~DNNLConvBlockedKernel();
};
#endif
