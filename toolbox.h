#ifndef _TOOLBOX
#define _TOOLBOX

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <asm/unistd.h>
#include <fstream>
#include <streambuf>
#include <sstream>

struct platform_info
{
    long num_logical_processors;
    long num_physical_processors_per_socket;
    long num_hw_threads_per_socket;
    unsigned int num_ht_threads; 
    unsigned int num_total_phys_cores;
    float tsc_ghz;
    unsigned long long max_bandwidth; 
    float gflops; // Giga Floating point operations per second
    int fmaspc;
};

class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1), m_tsc_ghz(0), m_fmaspc(0)
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
            std::string cpu_name;
            while(std::getline(cpuinfo_stream,cpuinfo_line,'\n')){
                if((m_num_physical_processors_per_socket == 0) && (cpuinfo_line.find("cpu cores") != std::string::npos)) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of physical cores per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_physical_processors_per_socket; 
                }
                if(cpuinfo_line.find("siblings") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_hw_threads_per_socket; 
                }

                if(cpuinfo_line.find("model") != std::string::npos) {
                    cpu_name = cpuinfo_line;
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find("@") + 1) ) >> m_tsc_ghz; 
                }
                
                // determine instruction set (AVX, AVX2, AVX512)
                if(m_fmaspc == 0) {
                  if (cpuinfo_line.find(" avx") != std::string::npos) {
                    m_fmaspc = 8;   // On AVX instruction set we have one FMA unit , width of registers is 256bits, so we can do 8 muls and adds on floats per cycle
                    if (cpuinfo_line.find(" avx2") != std::string::npos) {
                      m_fmaspc = 16;   // With AVX2 instruction set we have two FMA unit , width of registers is 256bits, so we can do 16 muls and adds on floats per cycle
                    }
                    if (cpuinfo_line.find(" avx512") != std::string::npos) {
                      m_fmaspc = 32;   // With AVX512 instruction set we have two FMA unit , width of registers is 512bits, so we can do 32 muls and adds on floats per cycle
                    }
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

            std::cout << "Platform:" << std::endl << "  " << cpu_name << std::endl 
                      << "  number of physical cores: " << m_num_total_phys_cores << std::endl; 
            ifs.close(); 

#endif
        }

    // Function computing percentage of theretical efficiency of HW capabilities
    float compute_theoretical_efficiency(unsigned long long start_time, unsigned long long end_time, unsigned long long num_fmas)
    {
      // Num theoretical operations
      // Time given is there
      return 100.0*num_fmas/((float)(m_num_total_phys_cores*m_fmaspc))/((float)(end_time - start_time));
    }

    void get_platform_info(platform_info& pi)
    {
       pi.num_logical_processors = m_num_logical_processors; 
       pi.num_physical_processors_per_socket = m_num_physical_processors_per_socket; 
       pi.num_hw_threads_per_socket = m_num_hw_threads_per_socket;
       pi.num_ht_threads = m_num_ht_threads;
       pi.num_total_phys_cores = m_num_total_phys_cores;
       pi.tsc_ghz = m_tsc_ghz;
       pi.max_bandwidth = m_max_bandwidth;
       pi.gflops = m_fmaspc*m_num_total_phys_cores*m_tsc_ghz;      
       pi.fmaspc = m_fmaspc;
    }
    private:
        long m_num_logical_processors;
        long m_num_physical_processors_per_socket;
        long m_num_hw_threads_per_socket;
        unsigned int m_num_ht_threads;
        unsigned int m_num_total_phys_cores;
        float m_tsc_ghz;
        short int m_fmaspc;
        unsigned long long m_max_bandwidth;
};

class ToolBox
{
  public:
    ToolBox(bool clear_cache) : clear_cache_(clear_cache), total_(0), llc_cache_linesize_(0), buf_(nullptr), size_of_floats_(128*1024*1024) {
      // Get LLC Cacheline size
      get_cacheline_size();

      if (clear_cache_) 
        allocate_buffer();
    }

    ~ToolBox() {
      if (clear_cache_) 
          free(buf_);
    }

  protected:
    void get_cacheline_size(void) {
      // Try cache level 4 linesize if it exists then
      // it is LLC
      llc_cache_linesize_ = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
      // If no Level4 exists then L3 should be LLC
      if(llc_cache_linesize_ == 0) {
        llc_cache_linesize_ = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
      }
      // If no Level3 exists then L2 should be LLC
      if(llc_cache_linesize_ == 0) {
        llc_cache_linesize_ = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
      }
      // TODO: no cache iin arch?
      assert(llc_cache_linesize_ != 0);

      std::cout << "LLC cache_linesize: " << llc_cache_linesize_ << std::endl;
    }

    void allocate_buffer() {
      // Get 512 MB for source and copy it to 512 MB dst. 
      // Intention is to copy more memory than it can be fead into cache 
      int ret = posix_memalign((void**)&buf_,
              llc_cache_linesize_,size_of_floats_*sizeof(float));
      if (ret != 0) {
        std::cout << "Allocation error of source buffer!" << std::endl;
        exit(-1);
      }
    }

    void overwrite_caches() {
      // Generate some random data 
      for(unsigned int i=0; i < size_of_floats_; ++i) {
        buf_[i] = i;
      } 
    }
  protected: 
    bool clear_cache_;
    unsigned long long total_; 
    size_t llc_cache_linesize_;
    float *buf_;
    size_t size_of_floats_;
};

class MemoryTraffic : public ToolBox 
{
 public:
  MemoryTraffic(bool clear_cache = true) : ToolBox(clear_cache), fd_(-1) {
    memset(&pe_, 0, sizeof(struct perf_event_attr));
    // HW conters LLC -> DRAM are not measuring prefetcher
    //pe_.type = PERF_TYPE_HARDWARE;
    //pe_.config = PERF_COUNT_HW_CACHE_MISSES; 
    pe_.type = PERF_TYPE_RAW;
    pe_.config = 0xfed15051; 
    pe_.disabled = 1;
    pe_.exclude_kernel = 1; // exclude events taking place in kernel
    pe_.exclude_hv = 1;     // Exclude hypervisor

    // Measure memory traffic for this process and its execution on
    // any/all cpus
    fd_ = perf_event_open(&pe_, 0, -1, -1, 0);
    if (fd_ == -1) {
      throw "ERROR opening leader : PERF_COUNT_HW_CACHE_MISSES";
    }
  }

  ~MemoryTraffic() {
    close(fd_);
  }

  inline void StartCounting(void) {
    // If we are to do cold caches then we overwrite existing
    // caches to make sure no kernel used data is in cache
    if (clear_cache_)
        overwrite_caches();
    ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    return;
  }

  inline void StopCounting(void) {
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    long long count = 0;
    auto num_reads = read(fd_, &count, sizeof(long long));

    if (num_reads == -1) {
      throw "ERROR reading : PMU DRAM counters";
    }
    
    // LLC miss * linesize is rough memory traffic
    total_ =  (count)*llc_cache_linesize_;


    std::cout << "Final: " << count << std::endl;

    // Returning value to the cout stream directly makes lots of memory movement 
    std::cout << "MemoryTraffic: " << total_ << std::endl;
  }

 protected:
  long perf_event_open(struct perf_event_attr*
    hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
  {
    int ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
     group_fd, flags);

    return ret;
  }

 private:
  int fd_;
  struct perf_event_attr pe_;
};

class Runtime : public ToolBox
{
 public:
  Runtime(unsigned long long tsc_ghz, bool clear_cache = true) : ToolBox(clear_cache), tsc_ghz_(tsc_ghz), start_(0) {}


  ~Runtime() {
    double total_s = total_ / tsc_ghz_ / 1000000000.0f;       
    std::cout << "Runtime: " << total_ << " [cycles] "<< total_s << " [s]"<< std::endl;
  }

  double GetMeasure() {
    double total_s = total_ / tsc_ghz_ / 1000000000.0f;       
    return total_s;
  }

  inline void Start() {
    // If we are to do cold caches then we overwrite existing
    // caches to make sure no kernel used data is in cache
    if (clear_cache_)
        overwrite_caches();

    start_ = __rdtsc();
  }

  inline void Stop() {
    total_ += __rdtsc() - start_;
  }
 
 private:
  unsigned long long tsc_ghz_;
  unsigned long long start_; 
};



#endif
