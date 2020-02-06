#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_eltwise_blocked_kernel.hpp>

REGISTER_KERNEL(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_relu>);
REGISTER_KERNEL_VARIANT(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_swish>, swish);
REGISTER_KERNEL_VARIANT(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_gelu>, gelu);

template<dnnl::algorithm algo>
DNNLEltwiseBlockedKernel<algo>::DNNLEltwiseBlockedKernel() : DNNLEltwiseKernel<algo,0,0>(false)
{
  this->mappings_.clear();
  this->mappings_[dnnl::algorithm::eltwise_relu] = "dnnl_blocked_relu";
  this->mappings_[dnnl::algorithm::eltwise_swish] = "dnnl_blocked_swish";
  this->mappings_[dnnl::algorithm::eltwise_gelu] = "dnnl_blocked_gelu";
}

template<dnnl::algorithm algo>
void DNNLEltwiseBlockedKernel<algo>::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Input desc
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

}

template<dnnl::algorithm algo>
void DNNLEltwiseBlockedKernel<algo>::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

template<dnnl::algorithm algo>
void DNNLEltwiseBlockedKernel<algo>::ShowInfo(bool cold_caches)
{

}



