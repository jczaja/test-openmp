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
#include "mkl.h"

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


int main()
{
	//printf("Hello OpenMP World!. Thread limit: %d\n",omp_get_thread_limit());

    //const int num_elements = 1000;

    //float myarray[num_elements];
    //float outarray[num_elements];
    //for_add_openmp2(num_elements,myarray,outarray);
    const int num_reps = 100000;

    const int sized = 100000;
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
      bottom_uns[i] = (float)i/sized + 2.0f;
      top[i] = 0.0f;
    }
    

    std::cout << "Hello SIMD openmp!" << std::endl;
    float sumseq = 0.0f;
    float sumsimd = 0.0f;
    float sumsimd2 = 0.0f;

    const int batch_size = 300;
    const int num_classes = 50;


    unsigned long long  t1;
    
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


    free(bottom_uns);
    free(top);



    std::cout << "softmax SEQ is : " << seqt/((float)2.5*1000000.0) << " ms" << std::endl;
//    std::cout << "Softmax SEQ = " << sumseq << std::endl;
//    std::cout << "softmax SIMD = " << sumsimd << std::endl;
    std::cout << "softmax SIMD is :" << simdt/(float)seqt << " of sequence time" << std::endl;

//    std::cout << "Softmax SEQ = " << sumseq << std::endl;
//    std::cout << "softmax SIMD2 = " << sumsimd2 << std::endl;
    std::cout << "softmax SIMD2 is :" << simd2t/(float)seqt << " of sequence time" << std::endl;
	return 0;
}
