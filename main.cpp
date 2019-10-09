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
#include "kernels/kernel.h"

#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

DEFINE_int32(num_reps, 1,
"Number of repetitions of computations to be performed");
DEFINE_int32(batch_size, 1,
"Batch size to be used for compuations");
DEFINE_int32(channel_size, 1,
"Channel size to be used");
DEFINE_int32(height, 1,
"Height to be used for compuations");
DEFINE_int32(width, 1,
"Width to be used for compuations");

DEFINE_string(algo, "max", "Name of algorithm to execute. Possible values: max, sum, softmax. Default: max");
DEFINE_bool(cputest, false, "Whether to show cpu capabilities");
DEFINE_bool(memtest, false, "Whether to perform memory throughput test");

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




void run_cpu_test( platform_info& pi)
{
  //TODO(jczaja): Implement benchmark eg. FMA's based reduction code
  std::cout << " Maximal Floating point operation per second: " << pi.gflops << " [GFLOPS/second]" << std::endl;
}


void run_mem_test(platform_info& pi)
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

  // Memory non-temporaral writes
  auto memory_nontemp_write = [&](char* dst, size_t total_size, int num_threads) {
    __m256i* varray = (__m256i*) dst;

    __m256i vals = _mm256_set1_epi32(1);
    size_t i;
    auto start_t = __rdtsc();
    #pragma omp parallel for num_threads(num_threads) if (num_threads > 1)
    for (i = 0; i < total_size / sizeof(__m256); i++) {
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("BEGIN NON-TEMP <---");
#     endif
      _mm256_stream_si256(&varray[i], vals);  // This generates the vmovntdq instruction on Brix (i7-4700R (AVX2, Fedora 21)
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("END NON-TEMP <---");
#     endif
    }
    return __rdtsc() - start_t;
  };

  // Writting memory as fast as possible
  auto memory_write = [&](char* dst, size_t total_size, int num_threads) {
      auto size_to_write = total_size/num_threads;
      auto start_t = __rdtsc();
      #pragma omp parallel for num_threads(num_threads) if (num_threads > 1)
      for(int i=0; i < num_threads; ++i) { 
          memset(dst + i*size_to_write, 2,size_to_write);
      }
      return __rdtsc() - start_t;
  };

  // Copying data as fast as possible
  auto memory_copy = [&](char* dst, char* src , size_t total_size, int num_threads) {
      auto size_to_copy = total_size/num_threads;
      auto start_t = __rdtsc();
      #pragma omp parallel for num_threads(num_threads) if (num_threads > 1)
      for(int i=0; i < num_threads; ++i) { 
          memcpy(dst + i*size_to_copy,src+i*size_to_copy,size_to_copy);
      }
      return __rdtsc() - start_t;
  };

  std::vector<unsigned long long> mem_nontemp_write_times;
  mem_nontemp_write_times.emplace_back( memory_nontemp_write((char*)dst, size_of_floats*sizeof(float), 1));
  mem_nontemp_write_times.emplace_back( memory_nontemp_write((char*)dst, size_of_floats*sizeof(float), 2));
  mem_nontemp_write_times.emplace_back( memory_nontemp_write((char*)dst, size_of_floats*sizeof(float), 4));
  auto mem_nontemp_write_t = *(std::min_element(mem_nontemp_write_times.begin(), mem_nontemp_write_times.end()));
  auto nontemp_write_throughput = size_of_floats*sizeof(float) / (mem_nontemp_write_t / ((float)pi.tsc_ghz));
  std::cout << " Memory Non-Temporal Write Throughput: " << nontemp_write_throughput << " [GB/s]" << std::endl;


  std::vector<unsigned long long> mem_write_times;
  mem_write_times.emplace_back( memory_write((char*)dst, size_of_floats*sizeof(float), 1));
  mem_write_times.emplace_back( memory_write((char*)dst, size_of_floats*sizeof(float), 2));
  mem_write_times.emplace_back( memory_write((char*)dst, size_of_floats*sizeof(float), 4));
  auto mem_write_t = *(std::min_element(mem_write_times.begin(), mem_write_times.end()));
  auto write_throughput = size_of_floats*sizeof(float) / (mem_write_t / ((float)pi.tsc_ghz));
  std::cout << " Memory Write Throughput: " << write_throughput << " [GB/s]" << std::endl;
  

  std::vector<unsigned long long> memcpy_times;
  memcpy_times.emplace_back( memory_copy((char*)dst, (char*)src, size_of_floats*sizeof(float), 1));
  memcpy_times.emplace_back( memory_copy((char*)dst, (char*)src, size_of_floats*sizeof(float), 2));

  auto memcpy_t = *(std::min_element(memcpy_times.begin(), memcpy_times.end()));

  // Data was read and then write so Q = Q_r + Q_w
  auto throughput = 2.0f*size_of_floats*sizeof(float) / (memcpy_t / ((float)pi.tsc_ghz));

  std::cout << " Memory Copy Throughput: " << throughput << " [GB/s]" << std::endl;

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

    nn_hardware_platform machine;
    platform_info pi;
    machine.get_platform_info(pi);

    // CPU thoughput test
    if (FLAGS_cputest) {
       run_cpu_test(pi);
       return 0;
    }

    // Memory thoughput test
    if (FLAGS_memtest) {
       run_mem_test(pi);
       return 0;
    }

    std::cout << "Num reps: " << FLAGS_num_reps << std::endl;
    std::cout << "Channel Size: " << FLAGS_channel_size << std::endl;
    std::cout << "Batch Size: " << FLAGS_batch_size << std::endl;
    std::cout << "Height: " << FLAGS_height << std::endl;
    std::cout << "Width: " << FLAGS_width << std::endl;

    Kernel(pi, FLAGS_batch_size, FLAGS_channel_size, FLAGS_height, FLAGS_width).Run(FLAGS_num_reps);
	return 0;
}
