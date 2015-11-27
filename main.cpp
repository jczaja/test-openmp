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
#include <valgrind/callgrind.h>
#include <cassert>
#include <cmath>

void axpy_auto(float *y, float a, float *x, float c, unsigned long m)
{
    printf("Fukcja: %s\n",__PRETTY_FUNCTION__);
    unsigned long j = 0;
//asm volatile lines are for assembly generation only
//asm volatile ("---> Before loop!");
    for(j = 0; j < m; ++j) {
        y[j] = a* x[j] + c;
    }
//asm volatile ("After loop! <---");
}
typedef float float8 __attribute__((vector_size(32))); 
void axpy_vector_ext(float *y, float a, float *x, float c, unsigned long m)
{
    printf("Fukcja: %s\n",__PRETTY_FUNCTION__);
    float8 a8 = {a,a,a,a,a,a,a,a};
    float8 c8 = {c,c,c,c,c,c,c,c};
    unsigned long j = 0;
//asm volatile lines are for assembly generation only
//asm volatile ("---> Before VECTOR loop!");
    for(j = 0; j < m; j+=8) {
       *( (float8*)&y[j]) = a8*  (*( (float8*)&x[j])) + c8;
    }
//asm volatile ("After VECTOR loop! <---");
}

void test_single_memset(unsigned int N)  
{
    std::vector<float> weight_diff_mt;
    weight_diff_mt.resize(N);
    
    const unsigned int num_loops = 100;

    auto t1 = __rdtsc();
    for(unsigned int i=0; i< num_loops; ++i)
    {
        memset( &weight_diff_mt[0], 0., N * sizeof(float) );
    }
    auto t2 = __rdtsc();

    std::cout << "---> test_single_memset(" << N <<" ) takes " << ((t2 - t1)/num_loops) << " RDTSC cycles\n" << std::endl;
}

void test_openmp_memset(unsigned int N)  
{
    std::vector<float> weight_diff_mt;
    weight_diff_mt.resize(N);
    
    const unsigned int num_loops = 100;

    int num_threads = omp_get_max_threads();
    unsigned long chunk_size = N/num_threads;

    auto t1 = __rdtsc();
    for(unsigned int i=0; i< num_loops; ++i)
    {
        #pragma omp parallel if(N > 8192) 
        {
            //printf("thread_id=%d chunk_size=%ld \n",omp_get_thread_num(),chunk_size);
            memset( &weight_diff_mt[0] + omp_get_thread_num()*chunk_size, 0., chunk_size*sizeof(float) );
        }

    }
    auto t2 = __rdtsc();

    std::cout << "---> test_openmp_memset(" << N <<" ) takes " << ((t2 - t1)/num_loops) << " RDTSC cycles\n" << std::endl;
}

void test_openmp_threads_startup(unsigned int N)
{
    std::vector<float> weight_diff_mt;
    weight_diff_mt.resize(N);
    
    const unsigned int num_loops = 100;

    int num_threads = omp_get_max_threads();
    unsigned long chunk_size = N/num_threads;

    auto t1 = __rdtsc();
    #pragma omp parallel 
    {
        for(unsigned int i=0; i< num_loops; ++i)
        {
            //printf("thread_id=%d chunk_size=%ld \n",omp_get_thread_num(),chunk_size);
            memset( &weight_diff_mt[0] + omp_get_thread_num()*chunk_size, 0., chunk_size*sizeof(float) );
        }

    }
    auto t2 = __rdtsc();


    auto t3 = __rdtsc();
    #pragma omp parallel 
    {
        for(unsigned int i=0; i< num_loops; ++i)
        {
            //printf("thread_id=%d chunk_size=%ld \n",omp_get_thread_num(),chunk_size);
            memset( &weight_diff_mt[0] + omp_get_thread_num()*chunk_size, 0., chunk_size*sizeof(float) );
        }

    }
    auto t4 = __rdtsc();


    std::cout << "---> test_openmp_threads_startup(" << N <<" ) takes first:" << (t2-t1) << " second:" << (t4-t3) << " diff: " << ( (t4-t3) / ((float)(t2-t1)) ) << " RDTSC cycles " << "diff[ms]" << (( (t2-t1) - (t4 - t3) )/3200000.0f) << std::endl;
}


void check_sum(std::vector<unsigned long>& candidate_sum, std::vector<unsigned long>& input )
{
  unsigned int num_threads = omp_get_max_threads();
  unsigned long chunk_size = input.size()/num_threads;

  std::vector<unsigned long> weight_diff(chunk_size,0);
    
    for (unsigned int t = 0; t < num_threads ; ++t) {
      for (unsigned int j = 0; j < chunk_size ; ++j) {
        weight_diff[j] += input[t * chunk_size + j];
      }
    }

    for(unsigned int j=0;j< chunk_size; ++ j) {
        if(weight_diff[j] != candidate_sum[j]) {
            throw std::runtime_error("Error: sum does not match");
        }
    }

}

void test_single_sum(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt;
  weight_diff_mt.resize(N);

  unsigned int num_threads = omp_get_max_threads();
  unsigned int chunk_size = N/num_threads;

  std::vector<unsigned long> weight_diff;
  weight_diff.resize(N/num_threads);

  auto t1 = __rdtsc();
    for (unsigned int t = 0; t < num_threads ; ++t) {
      for (unsigned int j = 0; j < chunk_size ; ++j) {
        weight_diff[j] += weight_diff_mt[t * chunk_size + j];
      }
    }
  auto t2 = __rdtsc();

  std::cout << "---> test_single_sum(" << N <<" ) takes              " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}

void test_openmp_for_atomic_sum(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt(N,0);
  // fill with some data the input buffer
  for(unsigned int i=0;i<N;++i)
  {
    weight_diff_mt[i] = i;
  }

  unsigned int num_threads = omp_get_max_threads();
  unsigned int chunk_size = N/num_threads;

  std::vector<unsigned long> weight_diff(N/num_threads,0);

  auto t1 = __rdtsc();

  //#pragma omp parallel for collapse(2) 
  #pragma omp parallel for
  for (unsigned int t = 0; t < num_threads ; ++t) {
      for (unsigned int j = 0; j < chunk_size ; ++j) {
          #pragma omp atomic
          weight_diff[j] += weight_diff_mt[t * chunk_size + j];
      }
  }
  auto t2 = __rdtsc();

  check_sum(weight_diff,weight_diff_mt);

  std::cout << "---> test_openmp_for_atomic_sum(" << N <<" ) takes   " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}

// TODO: Redukcja do zrobienia
void test_openmp_reduction_sum(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt(N,0);
  // fill with some data the input buffer
  for(unsigned int i=0;i<N;++i)
  {
    weight_diff_mt[i] = i;
  }

  unsigned int num_threads = omp_get_max_threads();
  unsigned int chunk_size = N/num_threads;

  std::vector<unsigned long> weight_diff(N/num_threads,0);

  // sekcja parallel i kazdy z watkow dostaje porcje do wykonania


  auto t1 = __rdtsc();

  //#pragma omp parallel for collapse(2) 
  #pragma omp parallel for
  for (unsigned int t = 0; t < num_threads ; ++t) {
      for (unsigned int j = 0; j < chunk_size ; ++j) {
          #pragma omp atomic
          //#pragma omp critical
          weight_diff[j] += weight_diff_mt[t * chunk_size + j];
      }
  }
  auto t2 = __rdtsc();

  check_sum(weight_diff,weight_diff_mt);

  std::cout << "---> test_openmp_atomic_sum(" << N <<" ) takes " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}

void test_openmp_parallel_sum(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt(N,0);
  // fill with some data the input buffer
  for(unsigned int i=0;i<N;++i)
  {
    weight_diff_mt[i] = i;
  }
  CALLGRIND_START_INSTRUMENTATION;
  unsigned int num_threads = omp_get_max_threads();

  std::vector<unsigned long> weight_diff(N/num_threads,0);

  unsigned int chunk_size = N/num_threads;
  unsigned int col_per_thread = chunk_size/num_threads; 
  auto t1 = __rdtsc();
  #pragma omp parallel  
  {
    for(unsigned int j=0; j<col_per_thread;++j) {
      for (unsigned int t = 0; t < num_threads ; ++t) {
          weight_diff[omp_get_thread_num()*col_per_thread + j] += weight_diff_mt[t * chunk_size + omp_get_thread_num()*col_per_thread + j];
      }
    }
    
    // Wez kazdy kolejny wateczek robiac
    unsigned int j = col_per_thread*num_threads + omp_get_thread_num();
    if(j < chunk_size)
    {
      for (unsigned int t = 0; t < num_threads ; ++t) {
          weight_diff[j] += weight_diff_mt[t * chunk_size + j];
      }
    }

  }
  auto t2 = __rdtsc();

  CALLGRIND_STOP_INSTRUMENTATION;
  check_sum(weight_diff,weight_diff_mt);

  std::cout << "---> test_openmp_parallel_sum(" << N <<" ) takes     " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}


void test_openmp_parallel_sum_2(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt(N,0);
  // fill with some data the input buffer
  for(unsigned int i=0;i<N;++i)
  {
    weight_diff_mt[i] = i;
  }

  unsigned int num_threads = omp_get_max_threads();

  std::vector<unsigned long> weight_diff(N/num_threads,0);

  unsigned int chunk_size = N/num_threads;
  unsigned int col_per_thread = chunk_size/num_threads; 
  unsigned int extra_cols = chunk_size - col_per_thread*num_threads;        
  unsigned int base_offset = (extra_cols)*(col_per_thread + 1) -extra_cols*col_per_thread;
  
  auto t1 = __rdtsc();
  #pragma omp parallel firstprivate(base_offset) firstprivate(col_per_thread) 
  {
    // Dla tych tid <0 .. chunk_size - extra_cols> cols_per
    if((unsigned int)omp_get_thread_num() < extra_cols )
    {
        base_offset = omp_get_thread_num()*(++col_per_thread);
    } else {
        base_offset += omp_get_thread_num()*col_per_thread;
    }

    for(unsigned int j=0; j<col_per_thread;++j) {
      for (unsigned int t = 0; t < num_threads ; ++t) {
          weight_diff[ base_offset + j] += weight_diff_mt[t * chunk_size + base_offset + j];
      }
    }
  }
  auto t2 = __rdtsc();

  check_sum(weight_diff,weight_diff_mt);

  std::cout << "---> test_openmp_parallel_sum_2(" << N <<" ) takes   " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}


void test_openmp_simd_sum(unsigned int N)
{
  std::vector<unsigned long> weight_diff_mt(N,0);
  // fill with some data the input buffer
  for(unsigned int i=0;i<N;++i)
  {
    weight_diff_mt[i] = i;
  }

  unsigned int num_threads = omp_get_max_threads();

  std::vector<unsigned long> weight_diff(N/num_threads,0);

  unsigned int chunk_size = N/num_threads;
  unsigned int col_per_thread = chunk_size/num_threads; 
  auto t1 = __rdtsc();
  #pragma omp parallel 
  {
    for(unsigned int j=0; j<col_per_thread;++j) {
  //    #pragma omp simd safelen(16)
      for (unsigned int t = 0; t < num_threads ; ++t) {
          weight_diff[omp_get_thread_num()*col_per_thread + j] += weight_diff_mt[t * chunk_size + omp_get_thread_num()*col_per_thread + j];
      }
    }
    
    // Wez kazdy kolejny wateczek robiac
    unsigned int j = col_per_thread*num_threads + omp_get_thread_num();
    if(j < chunk_size)
    {
    //  #pragma omp simd safelen(16)
      for (unsigned int t = 0; t < num_threads ; ++t) {
          weight_diff[j] += weight_diff_mt[t * chunk_size + j];
      }
    }

  }
  auto t2 = __rdtsc();
  check_sum(weight_diff,weight_diff_mt);

  std::cout << "---> test_openmp_simd_sum(" << N <<" ) takes         " << ((t2 - t1)) << " RDTSC cycles" << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void init_maxpool(std::vector<unsigned long>& top1 , std::vector<unsigned long>& top2 ,std::vector<unsigned long>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    assert(width_ >= kernel_w_);
    assert(height >= kernel_h_);

    bottom.resize(num_*channels_*width_*height_);
    
    //fill the array in with data 
    for(int i=0; i < num_*channels_*width_*height_; ++i)
    {
        bottom[i] = i % 99; //pseudo random 
    }

    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    top1.resize(num_*channels_*pooled_width_*pooled_height_);
    top2.resize(num_*channels_*pooled_width_*pooled_height_);
}

unsigned long long test_single_maxpool(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (unsigned long)-1; 
    }
    
    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    std::vector<int> top_mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    std::vector<int> mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    bool use_top_mask = true;

    unsigned long* bottom_data = &bottom[0];
    unsigned long* top_data = &top[0];

    int *top_mask = &top_mask_idx_[0]; 
    int *mask = &mask_idx_[0];

    unsigned int top_offset = pooled_width_*pooled_height_;
    unsigned int bottom_offset = width_*height_;

    auto t1 = __rdtsc();
    // The main loop
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ ;
            int wstart = pw * stride_w_;
            int hend = std::min(hstart + kernel_h_, height_);
            int wend = std::min(wstart + kernel_w_, width_);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = index;
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom_offset; 
        top_data += top_offset;
        if (use_top_mask) {
          top_mask += top_offset;
        } else {
          mask += top_offset;
        }
      }
    }
    auto t2 = __rdtsc();

    return (t2 - t1);
}

unsigned long long test_openmp_maxpool(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (unsigned long)-1; 
    }

    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    std::vector<int> top_mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    std::vector<int> mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    bool use_top_mask = true;


    auto t1 = __rdtsc();
    // The main loop
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        unsigned long top_offset = pooled_height_*pooled_width_*(c + n*channels_);
        unsigned long* bottom_data =  &bottom[0] + width_*height_*(c + n*channels_);
        unsigned long* top_data = &top[0] + top_offset;
        int* mask = NULL;                                                            // 
        int* top_mask = NULL;                                                      //
        if (use_top_mask) {                                                            //
          top_mask =  &top_mask_idx_[0] + top_offset;
        } else {                                                                       //       max_idx_ dims?
          mask = &mask_idx_[0] + top_offset; 
        }                                                                              //
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ ;
            int wstart = pw * stride_w_;
            int hend = std::min(hstart + kernel_h_, height_);
            int wend = std::min(wstart + kernel_w_, width_);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = index;
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
      }
    }

    auto t2 = __rdtsc();
    return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	printf("Hello OpenMP World!. Thread limit: %d\n",omp_get_thread_limit());

    //test_openmp_threads_startup(1024*1024);
//    printf("Fukcja: %s\n",__PRETTY_FUNCTION__);

    //TODO: make it work eg. parallel scenario for any number of data
    // here we make number of data equally divadeble
    // to get started

#if 0   // sums accumulating performance testing
    //for(unsigned int buff_size =  num_threads*num_threads*256; buff_size < num_threads*256*256*num_threads; buff_size += buff_size)
    for(unsigned int buff_size =  num_threads*1024; buff_size < num_threads*1024*1024*10; buff_size += buff_size)
    {
        test_single_sum(buff_size);
        //test_openmp_for_atomic_sum(buff_size);
        //test_openmp_parallel_sum_2(buff_size);
        test_openmp_parallel_sum(buff_size);
        //test_openmp_simd_sum(buff_size);
    }
#endif

// TODO: TUTAJ PISALEM!!!!!

#if 0
    for(int batch = 1; batch <= 256; batch+=32)
    {
        for(int width_ = 11; width_ <200; width_+=width_) {
            for(int height_ = 11; height_ <200; height_+=height_) {
                for(int channels = 1; channels <=64; channels+=channels) {
                    for(int kernel_w_ = 2; kernel_w_ <= 11; ++kernel_w_) {
                        for(int kernel_h_ = 2; kernel_h_ <= 11; ++kernel_h_) {
                            for(int stride_h_ = 1; stride_h_ <= 4; ++stride_h_) {
                                for(int stride_w_ = 1; stride_w_ <= 4; ++stride_w_) {
                                    test_single_maxpool(batch, channels, width_, height_, kernel_w_, kernel_h_, stride_w_, stride_h_ );
                                }
                            }
                        }   
                    }
                }
            }
        }
    }
#endif

    int batch = 32;
    int channels = 50;
    int width =  221;
    int height =  221;
    int kernel_w_ = 4;
    int kernel_h_ = 4; 
    int stride_w_ = 2;
    int stride_h_ = 2;

    std::vector<unsigned long> top1;
    std::vector<unsigned long> top2;
    std::vector<unsigned long> bottom;

    init_maxpool(top1,top2, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
    auto openmp_res = test_openmp_maxpool(top1, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
    auto single_res = test_single_maxpool(top2, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );

    if(std::equal(top1.begin(),top1.end(),top2.begin()) == false)
    {
        printf("BLAD porownania!\n");
        exit(-1);
    }
  
    printf("batch=%d channels=%d width=%d height=%d kernel_width=%d kernel_height=%d stride_w_=%d stride_h_=%d single_time=%llu openmp_time=%llu ratio(openmp/single)=%f ",
            batch,channels,width,height,kernel_w_,kernel_h_,stride_w_,stride_h_,single_res,openmp_res, (openmp_res/(float)single_res));
    if(openmp_res <= single_res) {
        printf("%c[1;32mOpenMP!\n", 27); // red
    } else {
        printf("%c[1;31mSingle!\n", 27); // red
    }
    printf("%c[1;37m\n", 27); // red

/*
    const unsigned long array_length = 1000000000;
    //unsigned long *array = new unsigned long[array_length];
    unsigned long *in_array = nullptr;
    unsigned long *out_array = nullptr;
    int err = posix_memalign((void**)&in_array, 64, sizeof(unsigned long) * array_length);
    if(err) {
        printf("Error: allocation failed!\n");
        return -1;
    }
    err = posix_memalign((void**)&out_array, 64, sizeof(unsigned long) * array_length);
    if(err) {
        printf("Error: allocation failed!\n");
        return -1;
    }
    for(unsigned long i=0; i < array_length; ++i) {
        in_array[i] = (unsigned long)i;
        out_array[i] = (unsigned long)i;
    }

    auto t1 = __rdtsc();
    axpy_auto(out_array,2.0f,in_array,1.0f,array_length);
    auto t2 = __rdtsc();
    // Add rdtsc checking of time of execution 

    printf("---> AXPY_auto of %ld seconds takes %llu RDTSC cycles\n", array_length, (t2 - t1) );
    

    auto t3 = __rdtsc();
    axpy_vector_ext(out_array,2.0f,in_array,1.0f,array_length);
    auto t4 = __rdtsc();
    // Add rdtsc checking of time of execution 

    printf("---> AXPY_vector_ext of %ld seconds takes %llu RDTSC cycles\n", array_length, (t4 - t3) );
*/

	return 0;
}
