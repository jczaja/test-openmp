#ifndef _DNNL_FP32_KERNEL
#define _DNNL_FP32_KERNEL

#include <string>
#include <memory>
#include <toolbox.h>
#include <dnnl.hpp>
#include <kernels/base_kernel.hpp>

template<bool inplace>
class DNNLLayerNormKernel : public BaseKernel {

 public:
  // Registration of kernel
  DNNLLayerNormKernel();

  // Initialization
  // Params: dimensions
  void Init(platform_info &pi, int n, int c, int h, int w);

  // cleaning up and printing result
  ~DNNLLayerNormKernel();

  void ShowInfo(bool cold_caches);

 protected:
   void RunSingle(void);
   void InitializeData(float* ptr, unsigned int sized);

 private:
   dnnl::engine eng_;
   dnnl::stream s_;
   std::unique_ptr<dnnl::memory> src_;
   std::unique_ptr<dnnl::memory> scale_shift_;
   std::unique_ptr<dnnl::memory> dst_;
   std::unique_ptr<dnnl::primitive> layer_norm_;
   std::unordered_map<int, dnnl::memory> layer_norm_args_;
};

#endif
