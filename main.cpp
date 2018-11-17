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
      entities[n] = result; 

      for (int c=0; c < num_classes; ++c) {
        out_data[n*num_classes+c] = in_data[n*num_classes+c] - entities[n];
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

#pragma omp declare simd uniform(ptr,num_classes) linear(n:1)
float simd2_sum(float* ptr, int n, int num_classes)
{
   float result = 0.0f;
    float* tmpptr = ptr + n*num_classes;
   for (int c=0; c < num_classes; ++c) {
     result += tmpptr[c];
   }
  return result;
}

#pragma omp declare simd uniform(out_ptr,in_ptr,num_classes) linear(n:1)
void simd2_max(float* out_ptr, const float* in_ptr, int n, int num_classes)
{
    const float* tmpptr = in_ptr + n*num_classes;
    float max = tmpptr[0];
     for (int c=0; c < num_classes; ++c) {
       if (tmpptr[c] > max) {
         max = tmpptr[c];
       }
     }
     float* tmpptr_out = out_ptr + n*num_classes;

     for (int c=0; c < num_classes; ++c) {
       tmpptr_out[c] = tmpptr[c] - max;
     }
}



void simd2_softmax(const float* X,
                  float* Y, const int batch_size, const int num_classes) {

    const float* in_data = X;
    float* out_data = Y;
    // 2D data. Batch x C
    std::vector<float> entities(batch_size);
#     ifdef GENERATE_ASSEMBLY
      asm volatile ("BEGIN SIMD2-MAX <---");
#     endif
    #pragma omp simd
    for (int n=0; n < batch_size; ++n) {
      simd2_max(&out_data[0],&in_data[0],n,num_classes); 
    }
#   ifdef GENERATE_ASSEMBLY
    asm volatile ("END SIMD2-MAX <---");
#   endif
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

    int liczba_int = int();
    float liczba_float = float();

    printf("Liczba_float: %f liczba_int %d\n",liczba_float, liczba_int);

    //const int num_elements = 1000;

    //float myarray[num_elements];
    //float outarray[num_elements];
    //for_add_openmp2(num_elements,myarray,outarray);
    const int num_reps = 1000000;


    std::vector<float> bottom_uns(100000);
    std::vector<float> top(100000,0);
    for(size_t i=0; i<bottom_uns.size(); ++i) {
      bottom_uns[i] = (float)i/bottom_uns.size() + 2.0f;
    }
    
    float sumseq = 0.0f;
    float sumsimd = 0.0f;
    float sumsimd2 = 0.0f;

    const int batch_size = 50;
    const int num_classes = 50;


    unsigned long long  t1;
    
    // Warmup eg. does not account
    for (int n=0; n < num_reps; ++n) {
//      seq_sum(sumseq,bottom_uns);
      seq_softmax(&bottom_uns[0], &top[0],batch_size,num_classes); 
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





//    std::cout << "Softmax SEQ = " << sumseq << std::endl;
//    std::cout << "softmax SIMD = " << sumsimd << std::endl;
    std::cout << "softmax SIMD is :" << simdt/(float)seqt << " of sequence time" << std::endl;

//    std::cout << "Softmax SEQ = " << sumseq << std::endl;
//    std::cout << "softmax SIMD2 = " << sumsimd2 << std::endl;
    std::cout << "softmax SIMD2 is :" << simd2t/(float)seqt << " of sequence time" << std::endl;
	return 0;
}
