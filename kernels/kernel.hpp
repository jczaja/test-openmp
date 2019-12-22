#ifndef _MYKERNEL
#define _MYKERNEL

#include <string>
#include <kernels/base_kernel.hpp>

class Kernel : public BaseKernel
{
 public:

  // Registration of kernel
  Kernel();

  // Initialization
  // Params: dimensions
  void Init(platform_info &pi, int n, int c, int h, int w);

  void ShowInfo(bool cold_caches);

  // cleaning up and printing result
  ~Kernel() {
    if (buffer_) {
      std::cout << "Computed sum: " << result_ << std::endl;
      free(buffer_);
      buffer_ = nullptr;
    }
  }
     
 protected:
   void RunSingle(void);

 private:
   unsigned int sized_;
   float *buffer_;
   float result_; 
};

#endif
