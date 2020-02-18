#ifndef _DNNL_FP32_BINARY_KERNEL
#define _DNNL_FP32_BINARY_KERNEL

#include <string>
#include <memory>
#include <toolbox.h>
#include <dnnl.hpp>
#include <kernels/base_kernel.hpp>

extern std::unordered_map<std::string, BaseKernel*> kernels;

template<dnnl::algorithm algo>
class DNNLBinaryKernel : public BaseKernel {

public:
  DNNLBinaryKernel(bool register_kernel = true)
  {
    mappings_.clear();
    mappings_[static_cast<int>(dnnl::algorithm::binary_add)] = "dnnl_nchw_binary_add";
    mappings_[static_cast<int>(dnnl::algorithm::binary_mul)] = "dnnl_nchw_binary_mul";

    // registering kernel should no happen
    // when derived class is calling this constructor
    if (register_kernel == true)
      kernels[mappings_[static_cast<int>(algo)]] = this;
  }

  // Initialization
  // Params: dimensions
  void Init(platform_info &pi, int n, int c, int h, int w);

  // cleaning up and printing result
  virtual ~DNNLBinaryKernel();

  void ShowInfo(bool cold_caches);

 protected:
  void RunSingle(void);
  void InitializeData(float* ptr, unsigned int sized);

 protected:
  std::unordered_map<int, std::string> mappings_;

 protected:
   dnnl::engine eng_;
   dnnl::stream s_;
   std::unique_ptr<dnnl::memory> src0_;
   std::unique_ptr<dnnl::memory> src1_;
   std::unique_ptr<dnnl::memory> dst_;
   std::unique_ptr<dnnl::primitive> binary_;
   std::unordered_map<int, dnnl::memory> binary_args_;
};
#endif
