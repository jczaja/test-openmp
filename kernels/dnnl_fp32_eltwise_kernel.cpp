#include <iostream>
#include <unordered_map>
#include <x86intrin.h>
#include<kernels/dnnl_fp32_eltwise_kernel.hpp>

REGISTER_KERNEL(DNNLEltwiseKernel<dnnl::algorithm::eltwise_relu COMMA 0 COMMA 0>);
//REGISTER_KERNEL(DNNLEltwiseKernel<dnnl::algorithm::eltwise_swish COMMA 0 COMMA 0>); // Swish Alpha to be adjusted

template<dnnl::algorithm algo, int alpha, int beta>
DNNLEltwiseKernel<algo, alpha, beta>::DNNLEltwiseKernel()
{
  // Register kernel
  kernels[std::string("dnnl_nchw_eltwise")] = this;
}

template<dnnl::algorithm algo, int alpha, int beta>
void DNNLEltwiseKernel<algo, alpha,beta>::Init(platform_info &pi, int n, int c, int h, int w)
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

  // Allocate output (same format as input)
  dst_.reset(new dnnl::memory(src_md, eng_)); 

  auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
               algo, src_md, (float)alpha, (float)beta); 
  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, eng_);
  eltwise_.reset(new dnnl::eltwise_forward(eltwise_pd));
  eltwise_args_[DNNL_ARG_SRC] = *src_;
  eltwise_args_[DNNL_ARG_DST] = *dst_;
}


template<dnnl::algorithm algo, int alpha, int beta>
void DNNLEltwiseKernel<algo, alpha,beta>::InitializeData(float* ptr, unsigned int sized) {
  // Init with some random data
  for(unsigned int i=0; i< sized; ++i) {
      ptr[i] = i%13;
  }
}

template<dnnl::algorithm algo, int alpha, int beta>
void DNNLEltwiseKernel<algo, alpha,beta>::ShowInfo(void)
{
  auto src_md = src_->get_desc();
  auto dims = src_md.data.dims;
  int n = dims[0];
  int c = dims[1];
  int h = dims[2];
  int w = dims[3];

  std::cout << std::endl << " DNNL NCHW eltwise " << n << "x" << c 
        << "x" << h << "x" << w << std::endl << std::endl <<
  "   batch Size: "<< n << std::endl <<
  "   channel size: "<< c << std::endl <<
  "   height: "<< h << std::endl <<
  "   width: "<< w << std::endl << std::endl;
}


template<dnnl::algorithm algo, int alpha, int beta>
DNNLEltwiseKernel<algo, alpha,beta>::~DNNLEltwiseKernel()
{
  if (src_ ) {
   std::cout << "DNNL NCHW eltwise " << " SRC First element: " << static_cast<float*>(src_->get_data_handle())[0] << std::endl;
  }
  if (dst_ ) {
   std::cout << "DNNL NCHW eltwise " << " DST First element: " << static_cast<float*>(dst_->get_data_handle())[0] << std::endl;
  }
}

template<dnnl::algorithm algo, int alpha, int beta>
inline void DNNLEltwiseKernel<algo, alpha,beta>::RunSingle(void)
{
  eltwise_->execute(s_,eltwise_args_);
}

template<dnnl::algorithm algo, int alpha, int beta>
void DNNLEltwiseKernel<algo, alpha,beta>::Run(int num_reps)
{
#ifdef MEMORY_TRAFFIC_COUNT
    auto mt = ToolBox(true); // Just overwritting caches
#endif
#ifdef RUNTIME_TEST
    auto rt = Runtime(tsc_ghz_,false);
#endif
    for(int n = 0; n< num_reps; ++n) {
#ifdef RUNTIME_TEST
      rt.Start();
#endif
      RunSingle();  // Single iteration execution
#ifdef RUNTIME_TEST
      rt.Stop();
#endif
    }
}


