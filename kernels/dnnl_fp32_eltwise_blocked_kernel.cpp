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
  this->tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  this->eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  this->s_ = dnnl::stream(this->eng_);

  // Input desc
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  // Create computational primitive
  auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
           algo, src_md, 0, 0); // TODO(jczaja): alpha Beta support
  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, this->eng_);

  // Allocate input
  this->src_.reset(new dnnl::memory(eltwise_pd.src_desc(), this->eng_)); 
  this->InitializeData(static_cast<float*>(this->src_->get_data_handle()),n*c*h*w);

  // Allocate output
  this->dst_.reset(new dnnl::memory(eltwise_pd.dst_desc(), this->eng_)); 

  this->eltwise_.reset(new dnnl::eltwise_forward(eltwise_pd));

  this->eltwise_.reset(new dnnl::eltwise_forward(eltwise_pd));
  this->eltwise_args_[DNNL_ARG_SRC] = *this->src_;  
  this->eltwise_args_[DNNL_ARG_DST] = *this->dst_;  
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
  auto src_md = this->src_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  std::cout << std::endl << " DNNL Blocked "<< this->mappings_[algo] << " " << n << "x" << c << "x" 
         << h << "x" << w << " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
        std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl;
}


template<dnnl::algorithm algo>
DNNLEltwiseBlockedKernel<algo>::~DNNLEltwiseBlockedKernel()
{
  if (this->src_ ) {
   std::cout << "DNNL " << this->mappings_[algo] << 
       " SRC First element: " << static_cast<float*>(this->src_->get_data_handle())[0] << std::endl;
  }
  if (this->dst_ ) {
   std::cout << "DNNL " << this->mappings_[algo] << 
       " DST First element: " << static_cast<float*>(this->dst_->get_data_handle())[0] << std::endl;
  }

}


