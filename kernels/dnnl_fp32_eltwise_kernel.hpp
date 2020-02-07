#ifndef _DNNL_FP32_ELTWISE_KERNEL
#define _DNNL_FP32_ELTWISE_KERNEL

#include <string>
#include <memory>
#include <toolbox.h>
#include <dnnl.hpp>
#include <kernels/base_kernel.hpp>

template<dnnl::algorithm algo , int alpha , int beta>
class DNNLEltwiseKernel : public BaseKernel {

 public:
  // Registration of kernel
  DNNLEltwiseKernel(bool register_kernel = true);

  // Initialization
  // Params: dimensions
  void Init(platform_info &pi, int n, int c, int h, int w);

  // cleaning up and printing result
  ~DNNLEltwiseKernel();

  void ShowInfo(bool cold_caches);

 protected:
  void RunSingle(void);
   void InitializeData(float* ptr, unsigned int sized);

 protected:
  std::unordered_map<int, std::string> mappings_;

 protected:
   dnnl::engine eng_;
   dnnl::stream s_;
   std::unique_ptr<dnnl::memory> src_;
   std::unique_ptr<dnnl::memory> dst_;
   std::unique_ptr<dnnl::primitive> eltwise_;
   std::unordered_map<int, dnnl::memory> eltwise_args_;
};
#endif
