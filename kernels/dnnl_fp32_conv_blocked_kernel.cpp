#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_conv_blocked_kernel.hpp>

REGISTER_KERNEL(DNNLConvBlockedKernel);

DNNLConvBlockedKernel::DNNLConvBlockedKernel() :
DNNLKernel(false)
{
  // Register kernel
  kernels[std::string("dnnl_blocked_conv")] = this;
}

void DNNLConvBlockedKernel::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Input desc
  dnnl::memory::dims src_tz = {n, c, h, w};
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  auto num_filters = NumF;
  dnnl::memory::dims weights_tz = {num_filters, c, HeightF, WidthF};
  dnnl::memory::dims bias_tz = {num_filters};
  dnnl::memory::dims strides = {4,4};
  dnnl::memory::dims padding = {0,0};

  auto weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  auto bias_md = dnnl::memory::desc(bias_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  auto oh = (h + 2*padding[0] - weights_tz[2])/strides[0] + 1;
  auto ow = (w + 2*padding[1] - weights_tz[3])/strides[1] + 1;
  dnnl::memory::dims dst_tz = {n, num_filters, oh, ow};
  auto dst_md = dnnl::memory::desc(dst_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

  // Create computational primitive
  auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md, strides, padding, padding);
  auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, eng_); 

  // Allocate input
  src_.reset(new dnnl::memory(conv_pd.src_desc(), eng_)); 
  this->InitializeData(static_cast<float*>(src_->get_data_handle()),n*c*h*w);

  // Allocate weights & bias
  weights_.reset(new dnnl::memory(conv_pd.weights_desc(), eng_));
  this->InitializeData(static_cast<float*>(weights_->get_data_handle()),
     weights_tz[0]*weights_tz[1]*weights_tz[2]*weights_tz[3]);

  bias_.reset(new dnnl::memory(conv_pd.bias_desc(), eng_)); 
  this->InitializeData(static_cast<float*>(bias_->get_data_handle()), bias_tz[0]);

  // Allocate output
  dst_.reset(new dnnl::memory(conv_pd.dst_desc(), eng_)); 

  conv_.reset(new dnnl::convolution_forward(conv_pd));
  conv_args_[DNNL_ARG_SRC] = *src_;  
  conv_args_[DNNL_ARG_WEIGHTS] = *weights_;  
  conv_args_[DNNL_ARG_BIAS] = *bias_;  
  conv_args_[DNNL_ARG_DST] = *dst_;  
}

void DNNLConvBlockedKernel::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

void DNNLConvBlockedKernel::ShowInfo(bool cold_caches)
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

  std::cout << std::endl << " DNNL Blocked Conv " << n << "x" << c << "x" 
         << h << "x" << w << " " << NumF << "x" << HeightF << "x" << WidthF << 
        " (" << (cold_caches == true ? "cold_caches" : "warm_caches")  << ")" <<
        std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl <<
  "   num_filters: " << NumF << std::endl <<
  "   filter height: " << HeightF << std::endl <<
  "   filter_width: " << WidthF << std::endl << 
  "   output channel size: "<< oc << std::endl <<
  "   output height: "<< oh << std::endl <<
  "   output width: "<< ow << std::endl << std::endl;
}
