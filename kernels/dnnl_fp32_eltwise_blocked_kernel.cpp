#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_eltwise_blocked_kernel.hpp>

REGISTER_KERNEL(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_relu>);
REGISTER_KERNEL_VARIANT(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_swish>, swish);
REGISTER_KERNEL_VARIANT(DNNLEltwiseBlockedKernel<dnnl::algorithm::eltwise_gelu>, gelu);

template<dnnl::algorithm algo>
DNNLEltwiseBlockedKernel<algo>::DNNLEltwiseBlockedKernel() : DNNLEltwiseKernel<algo>(false)
{
  mappings_.clear();
  mappings_[dnnl::algorithm::eltwise_relu] = "dnnl_blocked_relu";
  mappings_[dnnl::algorithm::eltwise_swish] = "dnnl_blocked_swish";
  mappings_[dnnl::algorithm::eltwise_gelu] = "dnnl_blocked_gelu";
}
