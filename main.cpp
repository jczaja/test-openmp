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
//#include <gperftools/profiler.h>
#include <unistd.h>
#include <sys/types.h>


#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>

//#ifndef __rdtsc 
//unsigned long long __rdtsc()
//{
  //return 1;
//}
//#endif

struct platform_info
{
    long num_logical_processors;
    long num_physical_processors_per_socket;
    long num_hw_threads_per_socket;
    unsigned int num_ht_threads; 
    unsigned int num_total_phys_cores;
    unsigned long long tsc;
    unsigned long long max_bandwidth; 
};

class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1), m_tsc(0), m_max_bandwidth(0)
        {
#ifdef __linux__
            m_num_logical_processors = sysconf(_SC_NPROCESSORS_ONLN);
        
            m_num_physical_processors_per_socket = 0;

            std::ifstream ifs;
            ifs.open("/proc/cpuinfo"); 

            // If there is no /proc/cpuinfo fallback to default scheduler
            if(ifs.good() == false) {
                m_num_physical_processors_per_socket = m_num_logical_processors;
                assert(0);  // No cpuinfo? investigate that
                return;   
            }
            std::string cpuinfo_content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::stringstream cpuinfo_stream(cpuinfo_content);
            std::string cpuinfo_line;
            while(std::getline(cpuinfo_stream,cpuinfo_line,'\n')){
                if(cpuinfo_line.find("cpu cores") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of physical cores per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_physical_processors_per_socket; 
                    break;
                }
                if(cpuinfo_line.find("siblings") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_hw_threads_per_socket; 
                }

                if(cpuinfo_line.find("model") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    float ghz_tsc = 0.0f;
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find("@") + 1) ) >> ghz_tsc; 
                    m_tsc = static_cast<unsigned long long>(ghz_tsc*1000000000.0f);
                    
                    // Maximal bandwidth is Xeon 68GB/s , Brix 25.8GB/s
                    if(cpuinfo_line.find("Xeon") != std::string::npos) {
                      m_max_bandwidth = 68000;  //68 GB/s      -- XEONE5
                    } 
                    
                    if(cpuinfo_line.find("i7-4770R") != std::string::npos) {
                      m_max_bandwidth = 25800;  //25.68 GB/s      -- BRIX
                    } 
                }

            }
            // There is cpuinfo, but parsing did not get quite right? Investigate it
            assert( m_num_physical_processors_per_socket > 0);
            assert( m_num_hw_threads_per_socket > 0);

            // Calculate how many threads can be run on single cpu core , in case of lack of hw info attributes assume 1
            m_num_ht_threads =  m_num_physical_processors_per_socket != 0 ? m_num_hw_threads_per_socket/ m_num_physical_processors_per_socket : 1;
            // calculate total number of physical cores eg. how many full Hw threads we can run in parallel
            m_num_total_phys_cores = m_num_hw_threads_per_socket != 0 ? m_num_logical_processors / m_num_hw_threads_per_socket * m_num_physical_processors_per_socket : 1;

            ifs.close(); 

#endif
        }
    void get_platform_info(platform_info& pi)
    {
       pi.num_logical_processors = m_num_logical_processors; 
       pi.num_physical_processors_per_socket = m_num_physical_processors_per_socket; 
       pi.num_hw_threads_per_socket = m_num_hw_threads_per_socket;
       pi.num_ht_threads = m_num_ht_threads;
       pi.num_total_phys_cores = m_num_total_phys_cores;
       pi.tsc = m_tsc;
       pi.max_bandwidth = m_max_bandwidth;
    }
    private:
        long m_num_logical_processors;
        long m_num_physical_processors_per_socket;
        long m_num_hw_threads_per_socket;
        unsigned int m_num_ht_threads;
        unsigned int m_num_total_phys_cores;
        unsigned long long m_tsc;
        unsigned long long m_max_bandwidth;
};





template <typename Dtype>
inline void apply(int n, Dtype src[], Dtype target[], Dtype  (*func)(Dtype)) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        target[i] = func(src[i]);
    }
}

template <typename Dtype>
inline Dtype add(Dtype val1)
{
    return (val1 + (Dtype)1);
} 

#define for_add(n,a,o) \
    apply(n,a,o,add);

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
void init_maxpool(std::vector<float>& top1 , std::vector<float>& top2 ,std::vector<float>& top3, std::vector<float>& top4, std::vector<float>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    assert(width_ >= kernel_w_);
    assert(height_ >= kernel_h_);

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
    top3.resize(num_*channels_*pooled_width_*pooled_height_);
    top4.resize(num_*channels_*pooled_width_*pooled_height_);

    #pragma omp parallel
    {
        #pragma omp single
        printf("Initializing OpenMP thread pool\n");
    }
}

void init_relu(std::vector<float>& top1 , std::vector<float>& top2 ,std::vector<float>& top3, std::vector<float>& top4, std::vector<float>&bottom, int num_, int channels_, int width_, int height_)
{
    bottom.resize(num_*channels_*width_*height_);
    
    //fill the array in with data 
    for(int i=0; i < num_*channels_*width_*height_; ++i)
    {
        bottom[i] = (i % 99) - 50; //pseudo random 
    }

    top1.resize(num_*channels_*width_*height_);
    top2.resize(num_*channels_*width_*height_);
    top3.resize(num_*channels_*width_*height_);
    top4.resize(num_*channels_*width_*height_);

    #pragma omp parallel
    {
        #pragma omp single
        printf("RELU: Initializing OpenMP thread pool\n");
    }
}

unsigned long long test_single_maxpool(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (float)-1; 
    }
    
    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    std::vector<int> top_mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    std::vector<int> mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    bool use_top_mask = true;

    float* bottom_data = &bottom[0];
    float* top_data = &top[0];

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

unsigned long long test_openmp_maxpool(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (float)-1; 
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
        float* bottom_data =  &bottom[0] + width_*height_*(c + n*channels_);
        float* top_data = &top[0] + top_offset;
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

unsigned long long test_openmp_maxpool_2(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (float)-1; 
    }

    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    std::vector<int> top_mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    std::vector<int> mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    bool use_top_mask = true;


    int output_channel_size = pooled_height_*pooled_width_;
    int intput_channel_size = width_*height_;
    auto t1 = __rdtsc();
    // The main loop
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        unsigned long top_offset = output_channel_size *(c + n*channels_);
        float* bottom_data =  &bottom[0] + intput_channel_size*(c + n*channels_);
        float* top_data = &top[0] + top_offset;
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
            int index_h =  hstart * width_;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = index_h + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = index;
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
              index_h +=  width_;
            }
          }
        }
      }
    }

    auto t2 = __rdtsc();
    return (t2 - t1);
}


unsigned long long test_openmp_maxpool_3(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_, int kernel_w_, int kernel_h_, int stride_w_, int stride_h_ )
{
    // Clear Top buffer
    for(auto& topek : top) {
        topek = (float)-1; 
    }

    int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_) / stride_h_)) + 1;
    int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_) / stride_w_)) + 1;

    std::vector<int> top_mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    std::vector<int> mask_idx_(num_*channels_*pooled_width_*pooled_height_,-1);
    bool use_top_mask = true;


    int output_channel_size = pooled_height_*pooled_width_;
    int intput_channel_size = width_*height_;
    auto t1 = __rdtsc();
    // The main loop
    int chunk_size = num_/omp_get_max_threads()/4 > 1 ? num_/omp_get_max_threads()/4 : 1 ; 
    printf("chunk_size=%d\n",chunk_size);
    //int chunk_size = 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none), shared(num_,channels_,output_channel_size,intput_channel_size,bottom,top,use_top_mask,top_mask_idx_,mask_idx_,stride_h_,stride_w_,height_,width_,kernel_h_,kernel_w_) \
shared(pooled_height_,pooled_width_) 
#endif
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        unsigned long top_offset = output_channel_size *(c + n*channels_);
        float* bottom_data =  &bottom[0] + intput_channel_size*(c + n*channels_);
        float* top_data = &top[0] + top_offset;
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
            int index_h =  hstart * width_;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = index_h + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = index;
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
              index_h +=  width_;
            }
          }
        }
      }
    }

    auto t2 = __rdtsc();
    return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_maxpooling(std::vector<float> top1,std::vector<float> top2,std::vector<float> top3,std::vector<float> top4,std::vector<float> bottom)
{
    int win_openmp_counter =0, win_openmp_2_counter = 0, win_openmp_3_counter = 0;

    const std::string log_name = std::string("gperf_log.") + std::to_string(getpid());
    //ProfilerStart(log_name.c_str());

    //for(int batch = 32; batch <= 256; batch+=32)
    int batch=32;
    {
        //for(int channels = 1; channels <=64; channels+=channels) {
        int channels = 3;
        {

//        for(int width = 11; width <260; width+=width) {
          int width = 220;
            int height = width;
            int kernel_w_ = 11;
            int kernel_h_ = 11;
            int stride_w_ = 2;
            int stride_h_ = 2; 
                    //for(int kernel_w_ = 2; kernel_w_ <= 11; ++kernel_w_) {
                        //for(int kernel_h_ = 2; kernel_h_ <= 11; ++kernel_h_) {
                            //for(int stride_h_ = 1; stride_h_ <= 4; stride_h_<<=1) {
                                //for(int stride_w_ = 1; stride_w_ <= 4; stride_w_<<=1) {

        init_maxpool(top1,top2, top3, top4, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
        auto openmp_3_res = test_openmp_maxpool_3(top4, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
        auto openmp_2_res = test_openmp_maxpool_2(top3, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );

    //#if 0
        auto single_res = test_single_maxpool(top2, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
        auto openmp_res = test_openmp_maxpool(top1, bottom, batch, channels, width, height, kernel_w_, kernel_h_ ,stride_w_, stride_h_ );
        if(std::equal(top1.begin(),top1.end(),top2.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }
      
        if(std::equal(top1.begin(),top1.end(),top3.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }

        if(std::equal(top1.begin(),top1.end(),top4.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }

        printf("batch=%d channels=%d width=%d height=%d kernel_width=%d kernel_height=%d stride_w_=%d stride_h_=%d single_time=%llu openmp_time=%llu ratio(openmp/single)=%f ",
                batch,channels,width,height,kernel_w_,kernel_h_,stride_w_,stride_h_,single_res,openmp_res, (openmp_res/(float)single_res));
        if(openmp_res <= single_res) {
            printf("%c[1;32mOpenMP!\n", 27); // green
        } else {
            printf("%c[1;31mSingle!\n", 27); // red
        }
        printf("%c[1;37m\n", 27); // white

        printf("batch=%d channels=%d width=%d height=%d kernel_width=%d kernel_height=%d stride_w_=%d stride_h_=%d single_time=%llu openmp_2_time=%llu ratio(openmp_2/single)=%f ",
                batch,channels,width,height,kernel_w_,kernel_h_,stride_w_,stride_h_,single_res,openmp_2_res, (openmp_2_res/(float)single_res));
        if(openmp_2_res <= single_res) {
            printf("%c[1;32mOpenMP!\n", 27); // red
        } else {
            printf("%c[1;31mSingle!\n", 27); // red
        }

        printf("batch=%d channels=%d width=%d height=%d kernel_width=%d kernel_height=%d stride_w_=%d stride_h_=%d single_time=%llu openmp_3_time=%llu ratio(openmp_3/single)=%f ",
                batch,channels,width,height,kernel_w_,kernel_h_,stride_w_,stride_h_,single_res,openmp_3_res, (openmp_3_res/(float)single_res));
        if(openmp_2_res <= single_res) {
            printf("%c[1;32mOpenMP!\n", 27); // red
        } else {
            printf("%c[1;31mSingle!\n", 27); // red
        }

        if((openmp_3_res > openmp_2_res) || (openmp_3_res > openmp_res))
        {
            printf("%c[1;33m----> OpenMP dynamic custom slower than dynamic default OpenMP or static OpenMP<-----!\n", 27); // red
        }

        if(openmp_3_res < openmp_2_res) {
            if(openmp_3_res <= openmp_res) {
                ++win_openmp_3_counter;
            } else {
                ++win_openmp_counter;
            }
        } else {
            if(openmp_2_res <= openmp_res) {
                ++win_openmp_2_counter;
            } else {
                ++win_openmp_counter;
            }
        }


        printf("%c[1;37m\n", 27); // red



                                //}
                            //}
                        //}   
                    //}
                //}
            //}
        }
    }

    //ProfilerStop();

    printf("%c[1;37m OpenMP wins: %d\n", 27,win_openmp_counter); // white
    printf("%c[1;37m OpenMP 2 wins: %d\n", 27,win_openmp_2_counter); // white
    printf("%c[1;37m OpenMP 3 wins: %d\n", 27,win_openmp_3_counter); // white

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_single_relu(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_)
{
  auto t1 = __rdtsc();
  const int count = num_*channels_*width_*height_;
  float negative_slope = 0.0f;
  for (int i = 0; i < count; ++i) {
    top[i] = std::max(bottom[i], 0.0f)
        + negative_slope * std::min(bottom[i], 0.0f);
  }
  auto t2 = __rdtsc();
  return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_openmp_relu(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_)
{
  auto t1 = __rdtsc();
  const int count = num_*channels_*width_*height_;
  float negative_slope = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < count; ++i) {
    top[i] = std::max(bottom[i], 0.0f)
        + negative_slope * std::min(bottom[i], 0.0f);
  }
  auto t2 = __rdtsc();
  return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_openmp_relu_2(std::vector<float>& top, std::vector<float>&bottom, int num_, int channels_, int width_, int height_)
{
  auto t1 = __rdtsc();
  const int count = num_*channels_*width_*height_;
  float negative_slope = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,16)
#endif
  for (int i = 0; i < count; ++i) {
    top[i] = std::max(bottom[i], 0.0f)
        + negative_slope * std::min(bottom[i], 0.0f);
  }
  auto t2 = __rdtsc();
  return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_relu(std::vector<float> top1,std::vector<float> top2,std::vector<float> top3,std::vector<float> top4,std::vector<float> bottom)
{
    int win_openmp_counter =0, win_openmp_2_counter = 0, win_openmp_3_counter = 0;

    for(int batch = 32; batch <= 256; batch+=batch)
    //int batch=32;
    {
        for(int channels = 1; channels <=4; channels+=channels) 
        //int channels = 3;
        {

//        for(int width = 11; width <260; width+=width) {
          int width = 220;
          int height = width;

        init_relu(top1,top2, top3, top4, bottom, batch, channels, width, height);
        auto openmp_3_res = (unsigned long long)100000000000;//test_openmp_maxpool_3(top4, bottom, batch, channels, width, height);
        auto openmp_2_res = test_openmp_relu_2(top3, bottom, batch, channels, width, height);

    //#if 0
        auto single_res = test_single_relu(top2, bottom, batch, channels, width, height);
        auto openmp_res = test_openmp_relu(top1, bottom, batch, channels, width, height);
        if(std::equal(top1.begin(),top1.end(),top2.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }
        if(std::equal(top1.begin(),top1.end(),top3.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }

#if 0      
        if(std::equal(top1.begin(),top1.end(),top4.begin()) == false)
        {
            printf("BLAD porownania!\n");
            exit(-1);
        }
#endif
        printf("batch=%d channels=%d width=%d height=%d single_time=%llu openmp_time=%llu ratio(openmp/single)=%f ",
                batch,channels,width,height,single_res,openmp_res, (openmp_res/(float)single_res));
        if(openmp_res <= single_res) {
            printf("%c[1;32mRELU OpenMP!\n", 27); // green
        } else {
            printf("%c[1;31mRELU Single!\n", 27); // red
        }
        printf("%c[1;37m\n", 27); // white

        //printf("batch=%d channels=%d width=%d height=%d single_time=%llu openmp_2_time=%llu ratio(openmp_2/single)=%f ",
                //batch,channels,width,height,single_res,openmp_2_res, (openmp_2_res/(float)single_res));
        //if(openmp_2_res <= single_res) {
            //printf("%c[1;32mRELU OpenMP!\n", 27); // red
        //} else {
            //printf("%c[1;31mRELU Single!\n", 27); // red
        //}
#if 0

        printf("batch=%d channels=%d width=%d height=%d single_time=%llu openmp_3_time=%llu ratio(openmp_3/single)=%f ",
                batch,channels,width,height,single_res,openmp_3_res, (openmp_3_res/(float)single_res));
        if(openmp_2_res <= single_res) {
            printf("%c[1;32mRELU OpenMP!\n", 27); // red
        } else {
            printf("%c[1;31mRELU Single!\n", 27); // red
        }

        if((openmp_3_res > openmp_2_res) || (openmp_3_res > openmp_res))
        {
            printf("%c[1;33m----> RELU OpenMP dynamic custom slower than dynamic default OpenMP or static OpenMP<-----!\n", 27); // red
        }
#endif

        if(openmp_3_res < openmp_2_res) {
            if(openmp_3_res <= openmp_res) {
                ++win_openmp_3_counter;
            } else {
                ++win_openmp_counter;
            }
        } else {
            if(openmp_2_res <= openmp_res) {
                ++win_openmp_2_counter;
            } else {
                ++win_openmp_counter;
            }
        }


        printf("%c[1;37m\n", 27); // red



                                //}
                            //}
                        //}   
                    //}
                //}
            //}
        }
    }

    printf("%c[1;37m RELU OpenMP wins: %d\n", 27,win_openmp_counter); // white
    printf("%c[1;37m RELU OpenMP 2 wins: %d\n", 27,win_openmp_2_counter); // white
    printf("%c[1;37m RELU OpenMP 3 wins: %d\n", 27,win_openmp_3_counter); // white
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_memset(std::vector<float>& vector_to_set)
{
  auto t1 = __rdtsc();
  memset((void*)&vector_to_set[0],4,vector_to_set.size()*sizeof(float));
  auto t2 = __rdtsc();
  return (t2 - t1);
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_memset_openmp(std::vector<float>& vector_to_set)
{
  auto t1 = __rdtsc();
  int num_threads = omp_get_max_threads();
  unsigned int chunk_size = vector_to_set.size()*sizeof(float)/num_threads; // TODO: Dokretka dla buforow nierownych wielkosci (ostatni wateczek wiecej cisnie
  #pragma omp parallel
  {  
    unsigned tid_offset = omp_get_thread_num() * chunk_size;
    memset((void*)&vector_to_set[0] + tid_offset,4,chunk_size);
  }
  auto t2 = __rdtsc();
  return (t2 - t1);
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void run_memset(std::vector<float> vector_to_set)
{
  std::cout << "Potential size of vector: " << vector_to_set.max_size() << std::endl;

  const unsigned long long brix_tsc_freq = 3200000000; //3.2GHz -- BRIX 
  const unsigned int brix_max_bandwith = 25800;  //25.8 GB/s    -- BRIX

  const unsigned long long xeone5_tsc_freq = 2300000000; //2.3GHz -- XEONE5 
  const unsigned int xeone5_max_bandwith = 68000;  //68 GB/s      -- XEONE5

  const unsigned int size_of_vector = 1024*1024*1024; // 1 GB
  const unsigned int throughput_unit = 1024*1024;   // We will give throughput in Megabytes 
  unsigned int throughput = 0;    // This is actual number of Memory units written (unit is as defined above)

  //vector_to_set.resize(vector_to_set.max_size()/10000);
  vector_to_set.resize(size_of_vector);
  std::cout << "Vector's new size: " << vector_to_set.size() << std::endl;

  auto openmp_memset = test_memset_openmp(vector_to_set); 
  auto single_memset = test_memset(vector_to_set);

  throughput =  (size_of_vector*sizeof(float)*brix_tsc_freq)/((float)single_memset)/throughput_unit;    
  std::cout << "Single Memset execution time: " << (single_memset*1000.0f) / (float)brix_tsc_freq << " ms  cycles: " << single_memset << std::endl; 
  std::cout << "Single Throughput: " << throughput << " MB. Ratio(throughput/brix_max_bandwith): " << (throughput/(float)brix_max_bandwith) << std::endl << std::endl;

  throughput =  (size_of_vector*sizeof(float)*brix_tsc_freq)/((float)openmp_memset)/throughput_unit;    
  std::cout << "Openmp Memset execution time: " << (openmp_memset*1000.0f) / (float)brix_tsc_freq << " ms  cycles: " << openmp_memset << std::endl; 
  std::cout << "Openmp Throughput: " << throughput << " MB. Ratio(throughput/brix_max_bandwith): " << (throughput/(float)brix_max_bandwith) << std::endl << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_memcpy(std::vector<float>& vector_dst,std::vector<float>& vector_src)
{
  auto t1 = __rdtsc();
  memcpy((void*)&vector_dst[0],(void*)&vector_src[0],vector_src.size()*sizeof(float));
  auto t2 = __rdtsc();
  return (t2 - t1);
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_copy(std::vector<float>& vector_dst,std::vector<float>& vector_src)
{
  auto t1 = __rdtsc();
  std::copy(vector_src.begin(),vector_src.end(),vector_dst.begin());
  auto t2 = __rdtsc();
  return (t2 - t1);
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long long test_memcpy_openmp(std::vector<float>& vector_dst,std::vector<float>& vector_src)
{
  auto t1 = __rdtsc();
  int num_threads = omp_get_max_threads();
  unsigned int chunk_size = vector_src.size()*sizeof(float)/num_threads; // TODO: Dokretka dla buforow nierownych wielkosci (ostatni wateczek wiecej cisnie
  #pragma omp parallel
  {
    unsigned tid_offset = omp_get_thread_num() * chunk_size;
    memcpy((void*)&vector_dst[0] + tid_offset,(void*)&vector_src[0] + tid_offset,chunk_size);
  }
  auto t2 = __rdtsc();
  return (t2 - t1);
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_memcpy(std::vector<float>& vector_dst,std::vector<float>& vector_src)
{
  std::cout << "Potential size of vector: " << vector_src.max_size() << std::endl;

  const unsigned long long brix_tsc_freq = 3200000000; //3.2GHz -- BRIX 
  const unsigned int brix_max_bandwith = 25800;  //25.8 GB/s    -- BRIX

  const unsigned long long xeone5_tsc_freq = 2300000000; //2.3GHz -- XEONE5 
  const unsigned int xeone5_max_bandwith = 68000;  //68 GB/s      -- XEONE5

  const unsigned int size_of_vector = 1024*1024*1024; // 1 GB
  const unsigned int throughput_unit = 1024*1024;   // We will give throughput in Megabytes 
  unsigned int throughput = 0;    // This is actual number of Memory units written (unit is as defined above)

  vector_src.resize(size_of_vector);
  vector_dst.resize(size_of_vector);
  std::cout << "Src Vector's new size: " << vector_src.size() << std::endl;
  std::cout << "Dst Vector's new size: " << vector_dst.size() << std::endl;

  auto single_copy = test_copy(vector_dst, vector_src);
  auto openmp_memcpy = test_memcpy_openmp(vector_dst, vector_src); 
  auto single_memcpy = test_memcpy(vector_dst, vector_src);

  throughput =  (2*size_of_vector*sizeof(float)*xeone5_tsc_freq)/((float)single_memcpy)/throughput_unit;    
  std::cout << "Single Memcpy execution time: " << (single_memcpy*1000.0f) / (float)xeone5_tsc_freq << " ms  cycles: " << single_memcpy << std::endl; 
  std::cout << "Single Throughput: " << throughput << " MB. Ratio(throughput/xeone5_max_bandwith): " << (throughput/(float)xeone5_max_bandwith) << std::endl << std::endl;

  throughput =  (2*size_of_vector*sizeof(float)*xeone5_tsc_freq)/((float)single_copy)/throughput_unit;    
  std::cout << "Single Memcpy execution time: " << (single_copy*1000.0f) / (float)xeone5_tsc_freq << " ms  cycles: " << single_copy << std::endl; 
  std::cout << "Single Throughput: " << throughput << " MB. Ratio(throughput/xeone5_max_bandwith): " << (throughput/(float)xeone5_max_bandwith) << std::endl << std::endl;

  throughput =  (2*size_of_vector*sizeof(float)*xeone5_tsc_freq)/((float)openmp_memcpy)/throughput_unit;    
  std::cout << "Openmp Memcpy execution time: " << (openmp_memcpy*1000.0f) / (float)xeone5_tsc_freq << " ms  cycles: " << openmp_memcpy << std::endl; 
  std::cout << "Openmp Throughput: " << throughput << " MB. Ratio(throughput/xeone5_max_bandwith): " << (throughput/(float)xeone5_max_bandwith) << std::endl << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
unsigned long long test_caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  auto t1 = __rdtsc();
    if (alpha == 0) {
      memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
      auto t2 = __rdtsc();
      return (t2 - t1);
    }
    for (int i = 0; i < N; ++i) {
      Y[i] = alpha;
    }
  auto t2 = __rdtsc();
  return (t2 - t1);
}

template <typename Dtype>
unsigned long long test_master_caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  unsigned long long t1 = __rdtsc();
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
      auto t2 = __rdtsc();
      return (t2 - t1);
  }
#ifdef _OPENMP
  #pragma omp parallel if (omp_in_parallel() == 0)
  #pragma omp for
#endif
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
  unsigned long long t2 = __rdtsc();
  return (t2 - t1);
}

template <typename Dtype>
unsigned long long test_openmp_caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  auto t1 = __rdtsc();
    if (alpha == 0) {
      memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
      auto t2 = __rdtsc();
      return (t2 - t1);
    }

    // If we are executing parallel region already then do not start another one
    // if also number of data to be processed is smaller than arbitrary:
    // threashold 4 cachelines per thread then no parallelization is to be made
    int threshold = omp_get_max_threads()*768;
    bool run_parallel = (omp_in_parallel() == 0) &&
                        (N >= threshold) ? true : false;
    if(run_parallel) {
  #ifdef _OPENMP
    #pragma omp parallel for 
  #endif
    for (int i = 0; i < N; ++i) {
      Y[i] = alpha;
    }
  } else {
    for (int i = 0; i < N; ++i) {
      Y[i] = alpha;
    }
  }
  auto t2 = __rdtsc();
  if(!run_parallel) {
  printf("---> SINGLE in OPENMP\n");
  }
  return (t2 - t1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void run_caffe_set(std::vector<Dtype> vector_to_set1, std::vector<Dtype> vector_to_set2)
{
  std::cout << "Potential size of vector: " << vector_to_set1.max_size() << std::endl;

  nn_hardware_platform machine;
  platform_info pi;

  machine.get_platform_info(pi);

  printf("Platform: TSC: %llu MAX_BANDWIDTH: %llu\n",pi.tsc,pi.max_bandwidth);

  const unsigned int max_size_of_vector = 1024*1024*1024; // 1 GB
  const unsigned int throughput_unit = 1024*1024;   // We will give throughput in Megabytes 
  unsigned int throughput = 0;    // This is actual number of Memory units written (unit is as defined above)

  const Dtype value_to_set = static_cast<Dtype>(4);

//  for(unsigned int size_of_vector=1024; size_of_vector<max_size_of_vector; size_of_vector += size_of_vector)
    unsigned int size_of_vector = 256000;
 {

    //vector_to_set.resize(vector_to_set.max_size()/10000);
    vector_to_set1.resize(size_of_vector);
    vector_to_set2.resize(size_of_vector);
    std::cout << "=====> Vector's new size: " << vector_to_set1.size() << std::endl;

    auto single_memset = test_master_caffe_set(size_of_vector, value_to_set, &vector_to_set1[0]);
    auto openmp_memset = test_openmp_caffe_set(size_of_vector, value_to_set, &vector_to_set2[0]); 

    throughput =  (size_of_vector*sizeof(float)*pi.tsc)/((float)single_memset)/throughput_unit;
    std::cout << "Single Caffe_set execution time: " << (single_memset*1000.0f) / (float)pi.tsc << " ms  cycles: " << single_memset << std::endl;
    std::cout << "Single Throughput: " << throughput << " MB. Ratio(throughput/max_bandwith): " << (throughput/(float)pi.max_bandwidth) << std::endl << std::endl;

    throughput =  (size_of_vector*sizeof(float)*pi.tsc)/((float)openmp_memset)/throughput_unit;
    std::cout << "Openmp Caffe_set execution time: " << (openmp_memset*1000.0f) / (float)pi.tsc << " ms  cycles: " << openmp_memset << std::endl;
    std::cout << "Openmp Throughput: " << throughput << " MB. Ratio(throughput/max_bandwith): " << (throughput/(float)pi.max_bandwidth) << std::endl << std::endl;

    if(openmp_memset <= single_memset) {
        printf("%c[1;32mCaffe_set OpenMP!\n", 27); // green
    } else {
        printf("%c[1;31mCaffe_set Single!\n", 27); // red
    }
    printf("%c[1;37m\n", 27); // white
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_openmp(void)
{
  int num_threads = omp_get_max_threads();
  printf("OMP_MAX_THREADS=%d\n",num_threads);
  omp_set_nested(1);
  #pragma omp parallel for num_threads(4)
  for(int i=0;i<5;++i) 
  {
    int tid = omp_get_thread_num();
    //#pragma omp parallel for //num_threads(8)
    #pragma omp parallel for num_threads(8)
    for(int j=0;j<7;++j) {
      int tid2 = omp_get_thread_num();
      printf("TID=%d TID2=%d\n",tid,tid2);
    }
  }
  
}

unsigned long process_task(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom, unsigned int start_idx, unsigned int count)
{
  unsigned long sumish = 0;
  for(unsigned int i = start_idx; i< start_idx + count ; ++i) {
    sumish += bottom[i]*bottom[i] + 2;
  } 
  return sumish;
}

unsigned long long process_data_tasks(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom, unsigned int num_jobs)
{
  auto t1 = __rdtsc();
  const unsigned int chunk_size = 1024*1024;  
  unsigned long sumish = 0;
  #pragma omp parallel
  #pragma omp single nowait
  for(unsigned int i = 0; i< num_jobs; i+=chunk_size) {
    #pragma omp task default(none) firstprivate(i) shared(bottom,top,sumish)
    {
    unsigned long ls =0;
    ls = process_task(top,bottom,i,chunk_size);
    #pragma omp atomic
    sumish += ls;
    }
  } 
  auto t2 = __rdtsc();
  std::cout << "TASKS: SUMMISH= " << sumish << std::endl;
  return (t2 - t1);
}

unsigned long long process_data_sequentialy(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom, unsigned int num_jobs)
{
  auto t1 = __rdtsc();
  unsigned long sumish = 0;
  for(unsigned int i = 0; i< num_jobs; ++i) {
    sumish += bottom[i]*bottom[i] + 2;
  } 
  auto t2 = __rdtsc();
  std::cout << "SEQ: SUMMISH= " << sumish << std::endl;
  return (t2 - t1);
}

void run_tasks(std::vector<unsigned long>& top, std::vector<unsigned long>&bottom)
{
  const int num_jobs = 1024*1024*100;

  nn_hardware_platform machine;
  platform_info pi;
  machine.get_platform_info(pi);

  for(int i = 0; i< num_jobs; ++i) {
    bottom.push_back(i);                  
  } 

  top.resize(num_jobs);          

  auto tasks_res = process_data_tasks(top,bottom,num_jobs);
  auto seq_res = process_data_sequentialy(top,bottom,num_jobs);

  std::cout << "Sequential execution time: " << (seq_res*1000.0f) / (float)pi.tsc << " ms  cycles: " << seq_res << std::endl;
  std::cout << "Tasks execution time: " << (tasks_res*1000.0f) / (float)pi.tsc << " ms  cycles: " << tasks_res << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	printf("Hello OpenMP World!. Thread limit: %d\n",omp_get_thread_limit());

    //test_openmp_threads_startup(1024*1024);
//    printf("Fukcja: %s\n",__PRETTY_FUNCTION__);

    float myarray[100];
    float outarray[100];
    for_add(100,myarray,outarray);

 //for(unsigned int i=0; i<100;++i) { outarray[i]=myarray[i] + 1; };# 706 "/home/jczaja/test-openmp/main.cpp"
 //for(unsigned int i=0; i<100;++i) { outarray[i]=myarray[i] + 1; };

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

    std::vector<float> top1;
    std::vector<float> top2;
    std::vector<float> top3;
    std::vector<float> top4;
    std::vector<float> bottom;



    std::vector<unsigned long> top_uns;
    std::vector<unsigned long> bottom_uns;
    run_tasks(top_uns,bottom_uns);

    //run_maxpooling(top1,top2,top3,top4,bottom);
    //run_relu(top1,top2,top3,top4,bottom);

    //run_memset(bottom);
    //run_memcpy(top1,bottom);
    //run_openmp();
    //
    
    //run_caffe_set<float>(top1,top2);

//#endif
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
