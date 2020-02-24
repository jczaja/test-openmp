#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_binary_blocked_kernel.hpp>

REGISTER_KERNEL(DNNLBinaryBlockedKernel<dnnl::algorithm::binary_add>);
REGISTER_KERNEL_VARIANT(DNNLBinaryBlockedKernel<dnnl::algorithm::binary_mul>, binary_mul);

template<dnnl::algorithm algo>
DNNLBinaryBlockedKernel<algo>::DNNLBinaryBlockedKernel() : DNNLBinaryKernel<algo>(false)
{
  this->mappings_.clear();
  this->mappings_[static_cast<int>(dnnl::algorithm::binary_add)] = "dnnl_blocked_binary_add";
  this->mappings_[static_cast<int>(dnnl::algorithm::binary_mul)] = "dnnl_blocked_binary_mul";

  // Register kernel
  kernels[this->mappings_[static_cast<int>(algo)]] = this;
}

template<dnnl::algorithm algo>
void DNNLBinaryBlockedKernel<algo>::Init(platform_info &pi, int n, int c, int h, int w)
{
  this->tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  this->eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  this->s_ = dnnl::stream(this->eng_);

  // Input desc
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32,
       pi.fmaspc == 32 ? dnnl::memory::format_tag::aBcd16b : 
       pi.fmaspc == 16 ? dnnl::memory::format_tag::aBcd8b :  dnnl::memory::format_tag::aBcd4b);

  auto binary_desc = dnnl::binary::desc(algo, src_md, src_md, src_md); 
  auto binary_pd = dnnl::binary::primitive_desc(binary_desc, this->eng_);  

  // Allocate inputs
  this->src0_.reset(new dnnl::memory(binary_pd.src_desc(), this->eng_)); 
  this->InitializeData(static_cast<float*>(this->src0_->get_data_handle()),n*c*h*w);
  this->src1_.reset(new dnnl::memory(binary_pd.src_desc(), this->eng_)); 
  this->InitializeData(static_cast<float*>(this->src1_->get_data_handle()),n*c*h*w);

  // Allocate output
  this->dst_.reset(new dnnl::memory(binary_pd.dst_desc(), this->eng_));

  this->binary_.reset(new dnnl::binary(binary_pd));
  this->binary_args_[DNNL_ARG_SRC_0] = *this->src0_;
  this->binary_args_[DNNL_ARG_SRC_1] = *this->src1_;
  this->binary_args_[DNNL_ARG_DST] = *this->dst_;
}

template<dnnl::algorithm algo>
void DNNLBinaryBlockedKernel<algo>::InitializeData(float* ptr, unsigned int sized) {
// No initializing data for Traffic counting
#ifndef MEMORY_TRAFFIC_COUNT
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
#endif
}

template<dnnl::algorithm algo>
void DNNLBinaryBlockedKernel<algo>::ShowInfo(bool cold_caches)
{
  auto src_md = this->src0_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  auto algorithm_info = this->mappings_[static_cast<int>(algo)];
  std::replace(algorithm_info.begin(), algorithm_info.end(), '_', ' ');

  std::cout << std::endl << " DNNL Blocked "<<  algorithm_info << " " << n << "x" << c
        << "x" << h << "x" << w << 
        " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
        std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl << std::endl;
}

template<dnnl::algorithm algo>
DNNLBinaryBlockedKernel<algo>::~DNNLBinaryBlockedKernel()
{
  if (this->src0_ ) {
   std::cout << "DNNL blocked binary " << " SRC First element: " << static_cast<float*>(this->src0_->get_data_handle())[0] << std::endl;
  }
  if (this->dst_ ) {
   std::cout << "DNNL blocked binary " << " DST First element: " << static_cast<float*>(this->dst_->get_data_handle())[0] << std::endl;
  }
}
