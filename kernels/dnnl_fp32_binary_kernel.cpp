#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_binary_kernel.hpp>

REGISTER_KERNEL(DNNLBinaryKernel<dnnl::algorithm::binary_add>);
REGISTER_KERNEL_VARIANT(DNNLBinaryKernel<dnnl::algorithm::binary_mul>, binary_mul);

template<dnnl::algorithm algo>
void DNNLBinaryKernel<algo>::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Allocate inputs 
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
  src0_.reset(new dnnl::memory(src_md, eng_)); 
  this->InitializeData(static_cast<float*>(src0_->get_data_handle()),n*c*h*w);
  src1_.reset(new dnnl::memory(src_md, eng_)); 
  this->InitializeData(static_cast<float*>(src1_->get_data_handle()),n*c*h*w);

  // Allocate output (same format as input)
  dst_.reset(new dnnl::memory(src_md, eng_)); 

  auto binary_desc = dnnl::binary::desc(algo, src_md, src_md, src_md); 
  auto binary_pd = dnnl::binary::primitive_desc(binary_desc, eng_);  
  binary_.reset(new dnnl::binary(binary_pd));
  binary_args_[DNNL_ARG_SRC_0] = *src0_;
  binary_args_[DNNL_ARG_SRC_1] = *src1_;
  binary_args_[DNNL_ARG_DST] = *dst_;
}

template<dnnl::algorithm algo>
void DNNLBinaryKernel<algo>::InitializeData(float* ptr, unsigned int sized) {
// No initializing data for Traffic counting
#ifndef MEMORY_TRAFFIC_COUNT
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
#endif
}

template<dnnl::algorithm algo>
void DNNLBinaryKernel<algo>::ShowInfo(bool cold_caches)
{
  auto src_md = src0_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  auto algorithm_info = this->mappings_[static_cast<int>(algo)];
  std::replace(algorithm_info.begin(), algorithm_info.end(), '_', ' ');

  std::cout << std::endl << " DNNL NCHW "<<  algorithm_info << " " << n << "x" << c
        << "x" << h << "x" << w << 
        " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
        std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl << std::endl;
}


template<dnnl::algorithm algo>
DNNLBinaryKernel<algo>::~DNNLBinaryKernel()
{
  if (src0_ ) {
   std::cout << "DNNL NCHW binary " << " SRC First element: " << static_cast<float*>(src0_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL NCHW binary " << " DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

template<dnnl::algorithm algo>
inline void DNNLBinaryKernel<algo>::RunSingle(void)
{
  binary_->execute(s_,binary_args_);
}


