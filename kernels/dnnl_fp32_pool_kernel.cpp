#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_pool_kernel.hpp>

REGISTER_KERNEL(DNNLPoolKernel<WidthF>);

template<unsigned int WP>
DNNLPoolKernel<WP>::DNNLPoolKernel(bool register_kernel)
{
  // registering kernel should no happen
  // when derived class is calling this constructor
  if (register_kernel == true)
    kernels[std::string("dnnl_nchw_pool_avg")] = this;
}


template<unsigned int WP>
void DNNLPoolKernel<WP>::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Allocate input
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
  src_.reset(new dnnl::memory(src_md, eng_)); 
  this->InitializeData(static_cast<float*>(src_->get_data_handle()),n*c*h*w);

  dnnl::memory::dims strides = {1,1};
  dnnl::memory::dims padding = {0,0};
  dnnl::memory::dims ksize = {WP,WP};

  // Allocate output
  auto oh = (h + 2*padding[0] - WP)/strides[0] + 1;
  auto ow = (w + 2*padding[1] - WP)/strides[1] + 1;
  dnnl::memory::dims dst_tz = {n, c, oh, ow};
  auto dst_md = dnnl::memory::desc(dst_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
  dst_.reset(new dnnl::memory(dst_md, eng_)); 

  // Create computational primitive
  auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference,
                   dnnl::algorithm::pooling_avg_exclude_padding, src_md, dst_md, strides, ksize, padding, padding);
  auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, eng_); 
  pool_.reset(new dnnl::pooling_forward(pool_pd));
  pool_args_[DNNL_ARG_SRC] = *src_;  
  pool_args_[DNNL_ARG_DST] = *dst_;  
}

template<unsigned int WP>
void DNNLPoolKernel<WP>::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

template<unsigned int WP>
void DNNLPoolKernel<WP>::ShowInfo(bool cold_caches)
{
  auto src_md = src_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  std::cout << std::endl << " DNNL NCHW Pool " << n << "x" << c << "x" 
         << h << "x" << w << " " << WP << "x" << WP << 
      " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
      std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl <<
  "   pool height: " << WP << std::endl <<
  "   pool_width: " << WP << std::endl; 
}

template<unsigned int WP>
DNNLPoolKernel<WP>::~DNNLPoolKernel()
{
  if (src_ ) {
   std::cout << "DNNL Pool Avg " << 
       " SRC First element: " << static_cast<float*>(src_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL pool avg " << 
       " DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

template<unsigned int WP>
inline void DNNLPoolKernel<WP>::RunSingle(void)
{
  pool_->execute(s_,pool_args_);
}
