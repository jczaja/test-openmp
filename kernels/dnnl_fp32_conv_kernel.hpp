#ifndef _DNNL_FP32_KERNEL
#define _DNNL_FP32_KERNEL

#include <string>
#include <memory>
#include <toolbox.h>
#include <dnnl.hpp>
#include <kernels/base_kernel.hpp>

template<unsigned int NF, unsigned int HF, unsigned int WF>
class DNNLKernel : public BaseKernel {

 public:
  // Registration of kernel
  DNNLKernel(bool register_kernel = true);

  // Initialization
  // Params: dimensions
  virtual void Init(platform_info &pi, int n, int c, int h, int w);

  // cleaning up and printing result
  virtual ~DNNLKernel();

  void ShowInfo(bool cold_caches);

  protected:
   void RunSingle(void);
   void InitializeData(float* ptr, unsigned int sized);

 protected:
   std::unique_ptr<dnnl::memory> src_;
   std::unique_ptr<dnnl::memory> weights_;
   std::unique_ptr<dnnl::memory> bias_;
   std::unique_ptr<dnnl::memory> dst_;
   std::unique_ptr<dnnl::primitive> conv_;
   std::unordered_map<int, dnnl::memory> conv_args_;
   dnnl::engine eng_;
   dnnl::stream s_;
};

#endif
