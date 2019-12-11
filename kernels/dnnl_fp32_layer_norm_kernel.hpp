#ifndef _DNNL_FP32_KERNEL
#define _DNNL_FP32_KERNEL

#include <string>
#include <memory>
#include <toolbox.h>
#include <dnnl.hpp>
#include <kernels/base_kernel.hpp>

class DNNLLayerNormKernel : public BaseKernel {

 public:
  // Registration of kernel
  DNNLLayerNormKernel();

  // Initialization
  // Params: dimensions
  void Init(platform_info &pi, int n, int c, int h, int w);

  // Measured Execution of kernel
  // params: number of repetitions to execute 
  // returns: total time in cycles as measured by TSC
  void Run(int num_reps);

  // cleaning up and printing result
  ~DNNLLayerNormKernel();

  void ShowInfo(void);

 protected:
   void RunSingle(void);
   void InitializeData(float* ptr, unsigned int sized);

 private:
   unsigned long long tsc_ghz_;
   dnnl::engine eng_;
   dnnl::stream s_;
   std::unique_ptr<dnnl::memory> src_;
   std::unique_ptr<dnnl::memory> mean_;
   std::unique_ptr<dnnl::memory> variance_;
   std::unique_ptr<dnnl::memory> dst_;
   std::unique_ptr<dnnl::primitive> layer_norm_;
   std::unordered_map<int, dnnl::memory> layer_norm_args_;
};

#endif
