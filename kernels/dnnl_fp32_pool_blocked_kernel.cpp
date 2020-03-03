#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_pool_blocked_kernel.hpp>

REGISTER_KERNEL(DNNLPoolBlockedKernel);


DNNLPoolBlockedKernel::DNNLPoolBlockedKernel() : DNNLPoolKernel(false)
{
  // Register kernel
  kernels[std::string("dnnl_blocked_pool_avg")] = this;
}

void DNNLPoolBlockedKernel::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Input desc
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32,
       pi.fmaspc == 32 ? dnnl::memory::format_tag::aBcd16b : 
       pi.fmaspc == 16 ? dnnl::memory::format_tag::aBcd8b :  dnnl::memory::format_tag::aBcd4b);

  dnnl::memory::dims strides = {1,1};
  dnnl::memory::dims padding = {0,0};
  dnnl::memory::dims ksize = {WidthF,WidthF};

  // Allocate output
  auto oh = (h + 2*padding[0] - WidthF)/strides[0] + 1;
  auto ow = (w + 2*padding[1] - WidthF)/strides[1] + 1;
  dnnl::memory::dims dst_tz = {n, c, oh, ow};
  auto dst_md = dnnl::memory::desc(dst_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  // Create computational primitive
  auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference,
                   dnnl::algorithm::pooling_avg_exclude_padding, src_md, dst_md, strides, ksize, padding, padding);
  auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, eng_); 

  // Allocate input
  src_.reset(new dnnl::memory(pool_pd.src_desc(), eng_)); 
  this->InitializeData(static_cast<float*>(src_->get_data_handle()),n*c*h*w);

  // Allocate output
  dst_.reset(new dnnl::memory(pool_pd.dst_desc(), eng_)); 

  pool_.reset(new dnnl::pooling_forward(pool_pd));
  pool_args_[DNNL_ARG_SRC] = *src_;  
  pool_args_[DNNL_ARG_DST] = *dst_;  
}

void DNNLPoolBlockedKernel::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

void DNNLPoolBlockedKernel::ShowInfo(bool cold_caches)
{
  auto src_md = src_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  std::cout << std::endl << " DNNL Blocked Pool avg " << n << "x" << c << "x" 
         << h << "x" << w << " " << WidthF << "x" << WidthF << 
      " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
      std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl <<
  "   pool height: " << WidthF << std::endl <<
  "   pool_width: " << WidthF << std::endl; 
}
