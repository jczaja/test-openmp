#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include <kernels/dnnl_fp32_layer_norm_kernel.hpp>

REGISTER_KERNEL(DNNLLayerNormKernel<false>);
REGISTER_KERNEL_VARIANT(DNNLLayerNormKernel<true>, inplace);

template<bool inplace>
DNNLLayerNormKernel<inplace>::DNNLLayerNormKernel()
{
  // Register kernel
  // TODO(jczaja): Make a const expr
  if(inplace == true) {
    kernels[std::string("dnnl_tnc_layer_norm_inplace")] = this;
  } else {
    kernels[std::string("dnnl_tnc_layer_norm")] = this;
  }
}

template<bool inplace>
void DNNLLayerNormKernel<inplace>::Init(platform_info &pi, int n, int c, int h, int w)
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

  // Alocate output
  // for inplace computation , input and output are the same buffer
  if (inplace == false) {
    dst_.reset(new dnnl::memory(src_md, eng_)); 
  } 

  // Create computational primitive
  auto layer_norm_desc = dnnl::layer_normalization_forward::desc(dnnl::prop_kind::forward_inference,
    src_md,
		0.0001f,
    dnnl::normalization_flags::use_scale_shift); 		
  
  // Alocate scale-shift
  auto layer_norm_pd = dnnl::layer_normalization_forward::primitive_desc(layer_norm_desc, eng_); 

  // Get Scale Shift format
  scale_shift_.reset(new dnnl::memory(layer_norm_pd.weights_desc(), eng_));
  this->InitializeData(static_cast<float*>(scale_shift_->get_data_handle()), 2*c);

  layer_norm_.reset(new dnnl::layer_normalization_forward(layer_norm_pd));
  layer_norm_args_[DNNL_ARG_SRC] = *src_;  
  layer_norm_args_[DNNL_ARG_SCALE_SHIFT] = *scale_shift_;
  if (inplace == false) {
    layer_norm_args_[DNNL_ARG_DST] = *dst_;  
  } else {
    layer_norm_args_[DNNL_ARG_DST] = *src_;  
  }
}

template<bool inplace>
void DNNLLayerNormKernel<inplace>::InitializeData(float* ptr, unsigned int sized)
{
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

template<bool inplace>
void DNNLLayerNormKernel<inplace>::ShowInfo(bool cold_caches)
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

template<bool inplace>
DNNLLayerNormKernel<inplace>::~DNNLLayerNormKernel()
{
  if (src_ ) {
   std::cout << "DNNL TNC Layer Norm " << " TNC SRC First element: " << static_cast<float*>(src_->get_data_handle())[0] << std::endl;
  }
  if (scale_shift_) {
   std::cout << "DNNL TNC Layer Norm " << " 2*C Scale_shift First element: " << static_cast<float*>(scale_shift_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL TNC Layer Norm " << " TNC DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

template<bool inplace>
void DNNLLayerNormKernel<inplace>::RunSingle(void)
{
  layer_norm_->execute(s_,layer_norm_args_);
}
