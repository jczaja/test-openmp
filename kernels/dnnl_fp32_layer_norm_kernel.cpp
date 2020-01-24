#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include <kernels/dnnl_fp32_layer_norm_kernel.hpp>

REGISTER_KERNEL(DNNLLayerNormKernel);

DNNLLayerNormKernel::DNNLLayerNormKernel()
{
  // Register kernel
  kernels[std::string("dnnl_tnc_layer_norm")] = this;
}

void DNNLLayerNormKernel::Init(platform_info &pi, int n, int c, int h, int w)
{
  tsc_ghz_ = pi.tsc_ghz;

  // Get CPU engine
  eng_ = dnnl::engine(dnnl::engine::kind::cpu,0);
  s_ = dnnl::stream(eng_);

  // Allocate input
  dnnl::memory::dims src_tz = {h*w,n,c}; // seq length, batch_size, channel_size
  auto src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::tnc);
  src_.reset(new dnnl::memory(src_md, eng_)); 
  this->InitializeData(static_cast<float*>(src_->get_data_handle()),n*c*h*w);
 
  // Alocate stats
  dnnl::memory::dims stats_tz = {h*w,n};
  auto stats_md = dnnl::memory::desc(stats_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::tn);
  mean_.reset(new dnnl::memory(stats_md, eng_));
  // ..mean
  this->InitializeData(static_cast<float*>(mean_->get_data_handle()),h*w*n);
  // .. variance
  variance_.reset(new dnnl::memory(stats_md, eng_));
  this->InitializeData(static_cast<float*>(variance_->get_data_handle()),h*w*n);

  // Alocate output
  dst_.reset(new dnnl::memory(src_md, eng_)); 

  // Create computational primitive
  
  auto layer_norm_desc = dnnl::layer_normalization_forward::desc(dnnl::prop_kind::forward_inference,
    src_md,
		stats_md,
		0.0001f,
    dnnl::normalization_flags::use_global_stats); 		
  
  auto layer_norm_pd = dnnl::layer_normalization_forward::primitive_desc(layer_norm_desc, eng_); 
  layer_norm_.reset(new dnnl::layer_normalization_forward(layer_norm_pd));
  layer_norm_args_[DNNL_ARG_SRC] = *src_;  
  layer_norm_args_[DNNL_ARG_MEAN] = *mean_;
  layer_norm_args_[DNNL_ARG_VARIANCE] = *variance_;
  layer_norm_args_[DNNL_ARG_DST] = *dst_;  
}

void DNNLLayerNormKernel::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

void DNNLLayerNormKernel::ShowInfo(bool cold_caches)
{
  auto src_md = src_->get_desc();
  auto dims = src_md.data.dims;
  int t = dims[0];
  int n = dims[1];
  int c = dims[2];

  std::cout << std::endl << " DNNL TNC Layer Norm " << t << "x" << n << "x" << c << 
  " (" << (cold_caches == true ? "cold caches" : "warm caches")  << ")" <<
  std::endl << std::endl <<
  "   sequence length: "<<  t << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl << std::endl;
}

DNNLLayerNormKernel::~DNNLLayerNormKernel()
{
  if (src_ ) {
   std::cout << "DNNL TNC Layer Norm " << " TNC SRC First element: " << static_cast<float*>(src_->get_data_handle())[0] << std::endl;
  }
  if (mean_) {
   std::cout << "DNNL TNC Layer Norm " << " TN MEAN First element: " << static_cast<float*>(mean_->get_data_handle())[0] << std::endl;
  }
  if (variance_) {
   std::cout << "DNNL TNC Layer Norm " << " TN VARIANCE First element: " << static_cast<float*>(variance_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL TNC Layer Norm " << " TNC DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

void DNNLLayerNormKernel::RunSingle(void)
{
  layer_norm_->execute(s_,layer_norm_args_);
}
