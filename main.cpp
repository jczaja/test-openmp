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
    for (int i = 0; i < bottom.size(); i++) {
      result += bottom[i];
    }
//    asm volatile ("END SIMD! <---");
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
    const int num_reps = 100000;


    std::vector<float> bottom_uns(100000);
    for(size_t i=0; i<bottom_uns.size(); ++i) {
      bottom_uns[i] = (float)i/bottom_uns.size() + 2.0f;
    }
    
    //run_tasks(top_uns,bottom_uns);
    float sumseq = 0.0f;
    float sumsimd = 0.0f;

    unsigned long long  t1;

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      simd_sum(sumsimd,bottom_uns);
    }
    auto simdt = __rdtsc() - t1;

    t1 = __rdtsc();
    for (int n=0; n < num_reps; ++n) {
      seq_sum(sumseq,bottom_uns);
    }
    auto seqt = __rdtsc() - t1;


    std::cout << "SUM SEQ = " << sumseq << std::endl;
    std::cout << "SUM SIMD = " << sumsimd << std::endl;
    std::cout << "OMP SIMD Reduction is :" << simdt/(float)seqt << " of sequence time" << std::endl;

	return 0;
}
