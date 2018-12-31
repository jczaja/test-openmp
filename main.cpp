#include <cstdio>
#include <time.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <float.h>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>


#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#ifdef USE_MKL
#include "mkl.h"
#endif
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
/////////////////////////////////////////
// We can assume that matrix dimensions are divisible by 8
struct maxFunc : public Xbyak::CodeGenerator {
    maxFunc()
{
#if defined(__x86_64__)
// calling convention RDI, RSI, RDX, RCX, R8, R9
// XMM0-7 (ints are passed that way)
//      RDI - Reference to Result
//      RSI - PTR to Array
//      RDX - Num classes 

// Regsters that need to be preserved: RBX,RBP, R12-R15

  Xbyak::util::Cpu current_cpu;
  if(current_cpu.has(Xbyak::util::Cpu::tAVX2)) {
    printf("AVX2 supported!\n");
  } else {
    printf("AVX2 not detected!\n");
  }

  mov (rcx,rdx);	
  push(rbx);
  shr (rcx,3);  // Divide by 8 (eight floats)
  shl (rdx,2);  // num of Output elements * size of float (4)
  shl (rcx,5);  // Trunc to 32 bytes 


	// Compute partial maximums
  vpbroadcastd(ymm0,ptr [rsi]);
  xor(rax,rax);				// Move offset for next 8 floating point values
  L("for_i");
    cmp(rax,rcx);
    jz("tail");
    vmovaps(ymm1,ptr [rsi + rax]);  // A
		add(rax,32);				// Move offset for next 8 floating point values
		vmaxps(ymm0,ymm0,ymm1);
    jmp("for_i");
  // Tail execution
  L("tail");
    sub(rdx,rcx);
    cmp(rdx,16);  
    jb("seq");
    vmovaps(xmm2,ptr [rsi + rax]);  // A
		add(rax,16);				// Move offset for next 4 floating point values
    sub(rdx,16);
		vperm2f128(ymm2,ymm2,ymm2,0);
		vmaxps(ymm0,ymm0,ymm2);  //partial maxes in ymm0
  L("seq");
	  cmp(rdx,0);
    jz("done");	
		vpbroadcastd(ymm2,ptr [rsi + rax]);
		vmaxps(ymm0,ymm0,ymm2);  //partial maxes in ymm0
    sub(rdx,4);
    add(rax,4);
    jmp("seq");
  L("done");
  // Get within shortlisted buffer maximum
	vperm2f128(ymm1,ymm0,ymm0,1);
  vmaxps(ymm0,ymm0,ymm1);  //partial maxes in ymm0
  vpermilps(xmm1,xmm0,0x1B);
  vmaxps(ymm0,ymm0,ymm1);  //partial maxes in ymm0
  vpermilps(xmm1,xmm0,1);
  vmaxps(ymm0,ymm0,ymm1);  //ymm0[0:31] contains global maximum
  vmovss(ptr[rdi],xmm0); // Result <-Max(X[.])
  pop(rbx);

  printf("Generating Max Value code\n");
#else
        printf("32bit not supported\n");
#endif
  ret();
}
};
////////////////////////


void seq_max(float& result, const float* X, int num_classes)
{
# ifdef GENERATE_ASSEMBLY
  asm volatile ("BEGIN MAX SEQUENCE! <---");
# endif
  result = X[0];
	for (int c=0; c < num_classes; ++c) {
		if (X[c] > result) {
			result = X[c];
		}
	}
# ifdef GENERATE_ASSEMBLY
  asm volatile ("END MAX SEQUENCE! <---");
# endif
}

void simd_max(float& result, const float* X, int num_classes)
{
# ifdef GENERATE_ASSEMBLY
  asm volatile ("BEGIN MAX SIMD! <---");
# endif
  result = X[0];
	#pragma omp simd reduction(max: result) aligned(X : 32)
	for (int c=0; c < num_classes; ++c) {
		if (X[c] > result) {
			result = X[c];
		}
	}
# ifdef GENERATE_ASSEMBLY
  asm volatile ("END MAX SIMD! <---");
# endif
}

void xbyak_max(float& result, std::vector<float>&bottom)
{

}



void seq_sum(float& result, std::vector<float>&bottom)
{
//    asm volatile ("BEGIN SEQUENCE! <---");
    result = 0.0f;
    for(unsigned int i = 0; i< bottom.size(); ++i) {
      result  += bottom[i];     
    } 

//    asm volatile ("END SEQUENCE! <---");
}

void simd_sum(float& result, std::vector<float>&bottom)
{
//    asm volatile ("BEGIN SIMD! <---");
    result = 0.0f;

    #pragma omp simd reduction(+: result)
    for (unsigned int i = 0; i < bottom.size(); i++) {
      result += bottom[i];
    }
//    asm volatile ("END SIMD! <---");
}


#ifdef USE_MKL
void seq_softmax(const float* X,
                  float* Y, const int batch_size, const int num_classes) {

    const float* in_data = X;
    float* out_data = Y;
    // 2D data. Batch x C
    std::vector<float> entities(batch_size);
    for (int n=0; n < batch_size; ++n) {
      auto result = in_data[n*num_classes];
      const float* tmpptr = &in_data[n*num_classes];
    //  #pragma omp simd reduction(max: result) aligned(tmpptr)
      for (int c=0; c < num_classes; ++c) {
        if (tmpptr[c] > result) {
          result = tmpptr[c];
        }
      }
      entities[n] = result; 

      for (int c=0; c < num_classes; ++c) {
        out_data[n*num_classes+c] = in_data[n*num_classes+c] - entities[n];
      }
    }
    vsExp(num_classes*batch_size, out_data, out_data);

    for (int n=0; n < batch_size; ++n) {
      auto result = 0.0f; 
      float* tmpptr = &out_data[n*num_classes];
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("BEGIN SEQUENCE! <---");
#     endif
 //     #pragma omp simd reduction(+: result) aligned(tmpptr)
      for (int c=0; c < num_classes; ++c) {
        result += tmpptr[c];
      }
      entities[n] = result; 
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("END SEQUENCE! <---");
#     endif
      cblas_sscal(num_classes, 1.0f/entities[n], &out_data[n*num_classes], 1);
    }
}
#endif




#ifdef USE_MKL
void simd_softmax(const float* X,
                  float* Y, const int batch_size, const int num_classes) {

    const float* in_data = X;
    float* out_data = Y;
    // 2D data. Batch x C
    std::vector<float> entities(batch_size);
    for (int n=0; n < batch_size; ++n) {
      auto result = in_data[n*num_classes];
      const float* tmpptr = &in_data[n*num_classes];
      //#pragma omp simd reduction(max: result) aligned(tmpptr)
      #pragma omp simd reduction(max: result)
      for (int c=0; c < num_classes; ++c) {
        if (tmpptr[c] > result) {
          result = tmpptr[c];
        }
      }

      for (int c=0; c < num_classes; ++c) {
        out_data[n*num_classes+c] = in_data[n*num_classes+c] - result;
      }
    }
    vsExp(num_classes*batch_size, out_data, out_data);

    for (int n=0; n < batch_size; ++n) {
      auto result = 0.0f; 
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("BEGIN SIMD! <---");
#     endif
      float* tmpptr = &out_data[n*num_classes];
      //#pragma omp simd reduction(+: result) aligned(tmpptr)
      #pragma omp simd reduction(+: result)
      for (int c=0; c < num_classes; ++c) {
        result += tmpptr[c];
      }
      entities[n] = result; 
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("END SIMD! <---");
#     endif
      cblas_sscal(num_classes, 1.0f/entities[n], &out_data[n*num_classes], 1);
    }
}
#endif

#pragma omp declare simd uniform(ptr,num_classes) linear(n:1) notinbranch aligned(ptr:32)
float simd2_sum(float* ptr, int n, int num_classes)
{
   float result = 0.0f;
    float* tmpptr = ptr + n*num_classes;
   for (int c=0; c < num_classes; ++c) {
     result += tmpptr[c];
   }
  return result;
}


#ifdef USE_MKL
void simd2_softmax(const float* X,
                  float* Y, const int batch_size, const int num_classes) {

    const float* in_data = X;
    float* out_data = Y;
    // 2D data. Batch x C
    std::vector<float> entities(batch_size);
    for (int n=0; n < batch_size; ++n) {
      auto result = in_data[n*num_classes];
      const float* tmpptr = &in_data[n*num_classes];
      //#pragma omp simd reduction(max: result) aligned(tmpptr)
      #pragma omp simd reduction(max: result) aligned(tmpptr:32)
      for (int c=0; c < num_classes; ++c) {
        if (tmpptr[c] > result) {
          result = tmpptr[c];
        }
      }

      for (int c=0; c < num_classes; ++c) {
        out_data[n*num_classes+c] = in_data[n*num_classes+c] - result;
      }
    }
    vsExp(num_classes*batch_size, out_data, out_data);

#     ifdef GENERATE_ASSEMBLY
      asm volatile ("BEGIN SIMD2 <---");
#     endif
    #pragma omp simd
    for (int n=0; n < batch_size; ++n) {
      //#pragma omp simd reduction(+: result) aligned(tmpptr)
      entities[n] = simd2_sum(&out_data[0],n,num_classes); 
    }
#   ifdef GENERATE_ASSEMBLY
    asm volatile ("END SIMD2 <---");
#   endif
    for (int n=0; n < batch_size; ++n) {
      cblas_sscal(num_classes, 1.0f/entities[n], &out_data[n*num_classes], 1);
    }
}
#endif

int main()
{
	//printf("Hello OpenMP World!. Thread limit: %d\n",omp_get_thread_limit());

    //const int num_elements = 1000;

    //float myarray[num_elements];
    //float outarray[num_elements];
    //for_add_openmp2(num_elements,myarray,outarray);
    const int num_reps = 1000000;

    const int sized = 1000000;
    float *bottom_uns, *top;
    
    int ret = posix_memalign((void**)&bottom_uns,32,sized*sizeof(float));
    if (ret != 0) {
      std::cout << "Allocation error of bottom!" << std::endl;
      exit(-1);
    }
    ret = posix_memalign((void**)&top,32,sized*sizeof(float));
    if (ret != 0) {
      std::cout << "Allocation error of top!" << std::endl;
      exit(-1);
    }
    
    for(size_t i=0; i<sized; ++i) {
      bottom_uns[i] = (float)i/sized + powf(-1.0f,(float)i)*2.0f;
      //bottom_uns[i] = (float)i;
      top[i] = 0.0f;
    }
    

    std::cout << "Hello SIMD openmp!" << std::endl;
    float sumseq = 0.0f;
    float sumsimd = 0.0f;
    float sumsimd2 = 0.0f;

    const int batch_size = 300;
    const int num_classes = 1007;


    unsigned long long  t1;
    
#ifdef USE_MKL
    // Warmup eg. does not account
    for (int n=0; n < num_reps; ++n) {
//      seq_sum(sumseq,bottom_uns);
      seq_softmax(bottom_uns, top,batch_size,num_classes); 
    }

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      simd2_softmax(&bottom_uns[0], &top[0],batch_size,num_classes); 
    }
    auto simd2t = __rdtsc() - t1;

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
//      simd_sum(sumsimd,bottom_uns);
      simd_softmax(&bottom_uns[0], &top[0],batch_size,num_classes); 
    }
    auto simdt = __rdtsc() - t1;


    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
//      seq_sum(sumseq,bottom_uns);
      seq_softmax(&bottom_uns[0], &top[0],batch_size,num_classes); 
    }
    auto seqt = __rdtsc() - t1;

    std::cout << "softmax SEQ is : " << seqt/((float)2.5*1000000.0) << " ms" << std::endl;
    std::cout << "softmax SIMD is :" << simdt/(float)seqt << " of sequence time" << std::endl;
    std::cout << "softmax SIMD2 is :" << simd2t/(float)seqt << " of sequence time" << std::endl;
#endif


    // Warmup eg. does not account
    float result1,result2,result3;

    maxFunc max_func;
		auto max_kernel = (void (*)(float& result, const float *x, int m))max_func.getCode();

		seq_max(result1,bottom_uns,num_classes); 

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      simd_max(result2,bottom_uns,num_classes); 
    }
    auto simd_t = __rdtsc() - t1;

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      max_kernel(result3,bottom_uns,num_classes); 
    }
    auto asm_t = __rdtsc() - t1;


    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      seq_max(result1,bottom_uns,num_classes); 
    }
    auto seq_t = __rdtsc() - t1;

		std::cout << "max SEQ = " << result1 << std::endl;
		std::cout << "max SIMD = " << result2 << std::endl;
		std::cout << "max ASM = " << result3 << std::endl;
    std::cout << "max SEQ is : " << seq_t/((float)2.4*1000000.0) << " ms" << std::endl;
    std::cout << "max SIMD is :" << simd_t/(float)seq_t << " of sequence time" << std::endl;
    std::cout << "max ASM is :" << asm_t/(float)seq_t << " of sequence time" << std::endl;

    free(bottom_uns);
    free(top);

	return 0;
}
