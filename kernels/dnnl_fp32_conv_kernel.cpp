#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_conv_kernel.hpp>

REGISTER_KERNEL(DNNLKernel<NumF COMMA HeightF COMMA WidthF>);

template<unsigned int NF, unsigned int HF, unsigned int WF>
DNNLKernel<NF, HF, WF>::DNNLKernel(bool register_kernel)
{
  // registering kernel should no happen
  // when derived class is calling this constructor
  if (register_kernel == true)
    kernels[std::string("dnnl_nchw_conv")] = this;
}


template<unsigned int NF, unsigned int HF, unsigned int WF>
void DNNLKernel<NF, HF, WF>::Init(platform_info &pi, int n, int c, int h, int w)
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

  // Allocate weights & bias
  auto num_filters = NF;
  dnnl::memory::dims weights_tz = {num_filters, c, HF, WF};
  dnnl::memory::dims bias_tz = {num_filters};
  dnnl::memory::dims strides = {1,1};
  dnnl::memory::dims padding = {0,0};

  auto weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw);
  weights_.reset(new dnnl::memory(weights_md, eng_)); 
  this->InitializeData(static_cast<float*>(weights_->get_data_handle()),
     weights_tz[0]*weights_tz[1]*weights_tz[2]*weights_tz[3]);

  auto bias_md = dnnl::memory::desc(bias_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  bias_.reset(new dnnl::memory(bias_md, eng_)); 
  this->InitializeData(static_cast<float*>(bias_->get_data_handle()), bias_tz[0]);

  // Allocate output
  auto oh = (h + 2*padding[0] - weights_tz[2])/strides[0] + 1;
  auto ow = (w + 2*padding[1] - weights_tz[3])/strides[1] + 1;
  dnnl::memory::dims dst_tz = {n, num_filters, oh, ow};
  auto dst_md = dnnl::memory::desc(dst_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
  dst_.reset(new dnnl::memory(dst_md, eng_)); 

  // Create computational primitive
  auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md, strides, padding, padding);
  auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, eng_); 
  conv_.reset(new dnnl::convolution_forward(conv_pd));
  conv_args_[DNNL_ARG_SRC] = *src_;  
  conv_args_[DNNL_ARG_WEIGHTS] = *weights_;  
  conv_args_[DNNL_ARG_BIAS] = *bias_;  
  conv_args_[DNNL_ARG_DST] = *dst_;  
}

template<unsigned int NF, unsigned int HF, unsigned int WF>
void DNNLKernel<NF, HF, WF>::InitializeData(float* ptr, unsigned int sized)
{
// No initializing data for Traffic counting
#ifndef MEMORY_TRAFFIC_COUNT
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
#endif
}

template<unsigned int NF, unsigned int HF, unsigned int WF>
void DNNLKernel<NF, HF, WF>::ShowInfo(bool cold_caches)
{
  auto src_md = src_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  auto dst_md = dst_->get_desc();
  dims = dst_md.data.dims;
  int oc = dims[1];
  int oh = dims[2];
  int ow = dims[3];

  std::cout << std::endl << " DNNL NCHW Conv " << n << "x" << c << "x" 
         << h << "x" << w << " " << NF << "x" << HF << "x" << WF << 
      " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
      std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl <<
  "   num_filters: " << NF << std::endl <<
  "   filter height: " << HF << std::endl <<
  "   filter_width: " << WF << std::endl << 
  "   output channel size: "<< oc << std::endl <<
  "   output height: "<< oh << std::endl <<
  "   output width: "<< ow << std::endl << std::endl;
}

template<unsigned int NF, unsigned int HF, unsigned int WF>
DNNLKernel<NF, HF, WF>::~DNNLKernel()
{
  if (src_ ) {
   std::cout << "DNNL conv " << 
       " SRC First element: " << static_cast<float*>(src_->get_data_handle())[0] << std::endl;
  }
  if (bias_ ) {
   std::cout << "DNNL conv " << 
       " BIAS First element: " << static_cast<float*>(bias_->get_data_handle())[0] << std::endl;
  }
  if (weights_ ) {
   std::cout << "DNNL conv " << 
       " WEIGHTS First element: " << static_cast<float*>(weights_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL conv " << 
       " DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

template<unsigned int NF, unsigned int HF, unsigned int WF>
inline void DNNLKernel<NF, HF, WF>::RunSingle(void)
{
  conv_->execute(s_,conv_args_);
}
