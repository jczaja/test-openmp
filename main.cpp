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
#include "gflags/gflags.h"

#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#ifdef USE_MKL
#include "mkl.h"
#endif
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

DEFINE_int32(num_reps, 1,
"Number of repetitions of computations to be performed");
DEFINE_int32(batch_size, 1,
"Batch size to be used for compuations");
DEFINE_int32(channel_size, 50,
"Dimm size of axe along which normalization takes place");
DEFINE_string(impl, "", "Name of implementation to execute. Possible values: seq, simd, jit. Default: Run all");
DEFINE_string(algo, "max", "Name of algorithm to execute. Possible values: max, sum, softmax. Default: max");
DEFINE_bool(memtest, false, "Whether to perform memory throughput test");
/////////////////////////////////////////
struct maxAFunc : public Xbyak::CodeGenerator {
    maxAFunc()
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

struct maxUFunc : public Xbyak::CodeGenerator {
    maxUFunc()
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
    vmovups(ymm1,ptr [rsi + rax]);  // A
		add(rax,32);				// Move offset for next 8 floating point values
		vmaxps(ymm0,ymm0,ymm1);
    jmp("for_i");
  // Tail execution
  L("tail");
    sub(rdx,rcx);
    cmp(rdx,16);  
    jb("seq");
    vmovups(xmm2,ptr [rsi + rax]);  // A
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
	#pragma omp simd reduction(max: result)
	for (int c=0; c < num_classes; ++c) {
		if (X[c] > result) {
			result = X[c];
		}
	}
# ifdef GENERATE_ASSEMBLY
  asm volatile ("END MAX SIMD! <---");
# endif
}

void seq_sum(float& result, const float* X, int num_classes)
{
//    asm volatile ("BEGIN SEQUENCE! <---");
    result = 0.0f;
    for(int i = 0; i< num_classes; ++i) {
      result  += X[i];     
    } 

//    asm volatile ("END SEQUENCE! <---");
}

void simd_sum(float& result, const float* X, int num_classes)
{
//    asm volatile ("BEGIN SIMD! <---");
    result = 0.0f;

    #pragma omp simd reduction(+: result)
    for (int i = 0; i < num_classes; i++) {
      result += X[i];
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

#pragma omp declare simd uniform(ptr,num_classes) linear(n:1) notinbranch
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

bool checkResults(std::vector<float>& results1, std::vector<float>& results2, float tolerance = 0.0f)
{
	bool consistency = true;
	for (unsigned int i=0; i<results1.size(); ++i) {
	 	consistency = consistency && (fabs(results1[i] - results2[i]) <= tolerance);
	}
  if (consistency == false) {
    
		for (unsigned int i=0; i<results1.size(); ++i) {
			printf("r1[%d]=%f r2[%d]=%f\n",i,results1[i],i,results2[i]);
		}
  }
	return consistency;
}

void run_sum_experiments(const float* bottom_uns)
{
    std::vector<float> result1(FLAGS_batch_size);
    std::vector<float> result2(FLAGS_batch_size);

    if (FLAGS_impl.empty()) {
			for (int b=0; b< FLAGS_batch_size; ++b) {
				seq_sum(result1[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size);
			} 
    }

    auto t1 = __rdtsc();
    if (FLAGS_impl.empty() || (FLAGS_impl.compare("simd") == 0)) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					simd_sum(result2[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size);
				} 
			}
    }
    auto simd_t = __rdtsc() - t1;

    t1 = __rdtsc();
    if (FLAGS_impl.empty() || (FLAGS_impl.compare("seq")==0)) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					seq_sum(result1[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size); 
				} 
			}
    }
    auto seq_t = __rdtsc() - t1;

    if (FLAGS_impl.empty()) {
			if (checkResults(result1,result2, 0.001) == false) {
				std::cout << "Error: Sum  for SIMD is inconsistent with SEQ" << std::endl;
				exit(-1);
			}

			std::cout << "sum SEQ is : " << seq_t/((float)2.4*1000000.0) << " ms" << std::endl;
			std::cout << "sum SIMD is :" << simd_t/(float)seq_t << " of sequence time" << std::endl;
	  }
}

void run_max_experiments(const float* bottom_uns)
{
		// First batch is aligned , all others are aligned if channel size is divisible by 8
		// Only execute when no specific algorithm is selected
		bool run_aligned = FLAGS_impl.empty() && ((FLAGS_channel_size % 8 == 0) || (FLAGS_batch_size == 1));

    // Warmup eg. does not account
    std::vector<float> result1(FLAGS_batch_size);
    std::vector<float> result2(FLAGS_batch_size);
    std::vector<float> result3(FLAGS_batch_size);
    std::vector<float> result4(FLAGS_batch_size);

    maxAFunc max_afunc;
    maxUFunc max_ufunc;
		auto max_akernel = (void (*)(float& result, const float *x, int m))max_afunc.getCode();
		auto max_ukernel = (void (*)(float& result, const float *x, int m))max_ufunc.getCode();

    if (FLAGS_impl.empty()) {
			for (int b=0; b< FLAGS_batch_size; ++b) {
				seq_max(result1[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size);
			} 
    }

    auto t1 = __rdtsc();
    if (FLAGS_impl.empty() || (FLAGS_impl.compare("simd") == 0)) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					simd_max(result2[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size);
				} 
			}
    }
    auto simd_t = __rdtsc() - t1;

    t1 = __rdtsc();
		if (run_aligned) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					max_akernel(result3[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size); 
				} 
			}
		}
    auto asma_t = __rdtsc() - t1;

    t1 = __rdtsc();
    if (FLAGS_impl.empty() || (FLAGS_impl.compare("jit") == 0)) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					max_ukernel(result4[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size); 
				} 
			}
    }
    auto asmu_t = __rdtsc() - t1;

    t1 = __rdtsc();
    if (FLAGS_impl.empty() || (FLAGS_impl.compare("seq")==0)) {
			for (int n=0; n < FLAGS_num_reps; ++n) {
				for (int b=0; b< FLAGS_batch_size; ++b) {
					seq_max(result1[b],&bottom_uns[b*FLAGS_channel_size],FLAGS_channel_size); 
				} 
			}
    }
    auto seq_t = __rdtsc() - t1;

    if (FLAGS_impl.empty()) {
			if (checkResults(result1,result2) == false) {
				std::cout << "Error: Max finding for SIMD is inconsistent with SEQ" << std::endl;
				exit(-1);
			}
			if ((run_aligned == true) && (checkResults(result1,result3) == false)) {
				std::cout << "Error: Max finding for aligned JIT is inconsistent with SEQ" << std::endl;
				exit(-1);
			}

			if (checkResults(result1,result4) == false) {
				std::cout << "Error: Max finding for unaligned JIT is inconsistent with SEQ" << std::endl;
				exit(-1);
			}

			std::cout << "max SEQ is : " << seq_t/((float)2.4*1000000.0) << " ms" << std::endl;
			std::cout << "max SIMD is :" << simd_t/(float)seq_t << " of sequence time" << std::endl;

			std::cout << "max unaligned JIT is :" << asmu_t/(float)seq_t << " of sequence time" << std::endl;
			if (run_aligned)
				std::cout << "max aligned JIT is :" << asma_t/(float)seq_t << " of sequence time" << std::endl;
	  }

}

void run_mem_test(void)
{
  // Get 512 MB for source and copy it to 512 MB dst. 
  // Intention is to copy more memory than it can be fead into cache 
  size_t size_of_floats = 128*1024*1024;
  float *src,*dst;
  int ret = posix_memalign((void**)&src,64,size_of_floats*sizeof(float));
  if (ret != 0) {
    std::cout << "Allocation error of source buffer!" << std::endl;
    exit(-1);
  }
  ret = posix_memalign((void**)&dst,64,size_of_floats*sizeof(float));
  if (ret != 0) {
    std::cout << "Allocation error of target buffer!" << std::endl;
    exit(-1);
  }

  // Generate data 
  for(unsigned int i=0; i < size_of_floats; ++i) {
    src[i] = i;
    dst[i] = 0.0f;
  } 

  // Copying data as fast as possible
  auto start_t = __rdtsc();
  memcpy(dst,src,size_of_floats*sizeof(float));
  auto memcpy_t = __rdtsc() - start_t;

  // Data was read and then write so Q = Q_r + Q_w
  float t_s = memcpy_t / ((float)2.4f);
  auto throughput = 2.0f*size_of_floats*sizeof(float) / t_s ;


  std::cout << "Q:" << (2.0f*size_of_floats*sizeof(float)) << std::endl;
  std::cout << "memcpy_t:" << memcpy_t << std::endl;
  std::cout << "t_s:" << t_s << std::endl;
  std::cout << "throughput:" << throughput << std::endl;

  std::cout << " Memory Throughput: " << throughput << " [GB/s]" << std::endl;

  free(src);
  free(dst);
}

int main(int argc, char** argv)
{
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Perform max & softmax computation.\n"
        "Usage:\n"
        "    test_openmp [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Memory thoughput test
		if (FLAGS_memtest) {
       run_mem_test();
       return 0;
    }

    const int sized = FLAGS_channel_size*FLAGS_batch_size;
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
    
    for(int i=0; i<sized; ++i) {
      bottom_uns[i] = (float)i/sized + powf(-1.0f,(float)i)*2.0f;
      //bottom_uns[i] = (float)i;
      top[i] = 0.0f;
    }
    

    std::cout << "Num reps: " << FLAGS_num_reps << std::endl;
    std::cout << "Channel Size: " << FLAGS_channel_size << std::endl;
    std::cout << "Batch Size: " << FLAGS_batch_size << std::endl;

		if (FLAGS_algo.compare("max") == 0)
			run_max_experiments(bottom_uns);
		if (FLAGS_algo.compare("sum") == 0)
			run_sum_experiments(bottom_uns);

    free(bottom_uns);
    free(top);

	return 0;
}
