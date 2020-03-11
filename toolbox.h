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
#include <iostream>

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
    bool is_xeon; 
};

class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1), m_tsc_ghz(0), m_fmaspc(0), m_is_xeon(false)
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
                    // If model name contains "Xeon" then we Assume Intel(R) Xeon platform
                    m_is_xeon = cpuinfo_line.find("Xeon") != std::string::npos;
                    
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
       pi.is_xeon = m_is_xeon; 
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
        bool m_is_xeon;
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
  MemoryTraffic(bool clear_cache = true) : ToolBox(clear_cache), fdr_(-1), fdw_(-1), count_data_reads_(0), count_data_writes_(0) {
    memset(&pe_, 0, sizeof(struct perf_event_attr));
    pe_.type = 12; // Desktop data_reads and data_writes
    pe_.config = 1; // 1 - data_reads, 2 - data_writes 
    pe_.disabled = 1;
    pe_.inherit = 1;
    pe_.exclude_kernel = 0; // Include Kernel events
    pe_.exclude_hv = 0;     // Include hypervisor

    fdr_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    // Now modify config to have data_writes counted    
    pe_.config = 2;
    fdw_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
  }

  virtual ~MemoryTraffic() {
    close(fdr_);
    close(fdw_);
  }

  virtual void StartCounting(void) {
    // If we are to do cold caches then we overwrite existing
    // caches to make sure no kernel used data is in cache
    if (clear_cache_)
        overwrite_caches();

    auto num_reads = read(fdr_, &count_data_reads_, sizeof(long long));
    if (num_reads == -1) {
      throw "ERROR reading : PMU DRAM counters";
    }
    std::cout << "data_reads: " << count_data_reads_ << " [64 bytes blocks]" << std::endl;
    num_reads = read(fdw_, &count_data_writes_, sizeof(long long));
    if (num_reads == -1) {
      throw "ERROR reading : PMU DRAM counters";
    }
    std::cout << "data_writes: " << count_data_writes_ << " [64 bytes blocks]" << std::endl;

    ioctl(fdr_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw_, PERF_EVENT_IOC_ENABLE, 0);
    return;
  }

  virtual void StopCounting(void) {
    ioctl(fdr_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw_, PERF_EVENT_IOC_DISABLE, 0);
    long long countr = 0;
    long long countw = 0;
    auto num_reads = read(fdr_, &countr, sizeof(long long));
    if (num_reads == -1) {
      throw "ERROR reading : PMU DRAM counters";
    }
    auto num_writes = read(fdw_, &countw, sizeof(long long));
    if (num_writes == -1) {
      throw "ERROR reading : PMU DRAM counters";
    }
    
    // IMC transfer * linesize is rough memory traffic
    auto data_reads_mib =  (countr - count_data_reads_)*llc_cache_linesize_/1024/1024;
    auto data_writes_mib =  (countw - count_data_writes_)*llc_cache_linesize_/1024/1024;

    std::cout << "MemoryTraffic: " << data_reads_mib << " MiB data_reads" << std::endl;
    std::cout << "MemoryTraffic: " << data_writes_mib << " MiB data_writes" << std::endl;
  }

 protected:
  long perf_event_open(struct perf_event_attr*
    hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
  {
    int ret;
    if((ret=syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags)) == -1 ) {
      throw "ERROR opening leader : Kernel PMU IMC";
    }
    return ret;
  }

 protected:
  int fdr_;
  int fdw_;
  long long count_data_reads_;
  long long count_data_writes_;
  struct perf_event_attr pe_;
};

class XeonMemoryTraffic : public MemoryTraffic 
{

 public:
  XeonMemoryTraffic(bool clear_cache = true) : MemoryTraffic(clear_cache), 
   fdr1_(-1), fdw1_(-1), fdr2_(-1), fdw2_(-1),fdr3_(-1), fdw3_(-1),
    fdr4_(-1), fdw4_(-1),fdr5_(-1), fdw5_(-1), 
 count_cas_read0_(0), count_cas_read1_(0), count_cas_read2_(0),
 count_cas_read3_(0), count_cas_read4_(0), count_cas_read5_(0),
 count_cas_write0_(0), count_cas_write1_(0), count_cas_write2_(0),
 count_cas_write3_(0), count_cas_write4_(0), count_cas_write5_(0)
 {
    memset(&pe_, 0, sizeof(struct perf_event_attr));
    pe_.config = 772; // 772 - cas_count_read, 3076 - cas_count_write
    pe_.disabled = 1;
    pe_.inherit = 1;
    pe_.exclude_kernel = 0; // Include Kernel events
    pe_.exclude_hv = 0;     // Include hypervisor

    // Intel Xeon Read IMC <0-5> read counters
    pe_.type = 14; // Intel Xeon cas_count_read and cas_count_write goes from 14 to 19
    fdr_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 15; // Intel Xeon cas_count_read IMC 1
    fdr1_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 16; // Intel Xeon cas_count_read IMC 2
    fdr2_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 17; // Intel Xeon cas_count_read IMC 3
    fdr3_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 18; // Intel Xeon cas_count_read IMC 4
    fdr4_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 19; // Intel Xeon cas_count_read IMC 5
    fdr5_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);

    // Now modify config to have data_writes counted    
    pe_.config = 3076;
    pe_.type = 14; // Intel Xeon cas_count_read and cas_count_write goes from 14 to 19
    fdw_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 15; // Intel Xeon cas_count_write IMC 1
    fdw1_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 16; // Intel Xeon cas_count_write IMC 2
    fdw2_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 17; // Intel Xeon cas_count_write IMC 3
    fdw3_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 18; // Intel Xeon cas_count_write IMC 4
    fdw4_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);
    pe_.type = 19; // Intel Xeon cas_count_write IMC 5
    fdw5_ = perf_event_open(&pe_, -1, 0, -1, PERF_FLAG_FD_CLOEXEC);


  }

  virtual ~XeonMemoryTraffic() {
    close(fdr_);
    close(fdw_);
    close(fdr1_);
    close(fdw1_);
    close(fdr2_);
    close(fdw2_);
    close(fdr3_);
    close(fdw3_);
    close(fdr4_);
    close(fdw4_);
    close(fdr5_);
    close(fdw5_);
  }

  virtual void StartCounting(void) {
    // If we are to do cold caches then we overwrite existing
    // caches to make sure no kernel used data is in cache
    if (clear_cache_)
        overwrite_caches();
    ioctl(fdr_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdr1_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdr2_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdr3_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdr4_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdr5_, PERF_EVENT_IOC_RESET, 0);

    ioctl(fdw_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdw1_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdw2_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdw3_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdw4_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fdw5_, PERF_EVENT_IOC_RESET, 0);

    count_cas_read0_ = get_pmu_value(fdr_);
    count_cas_read1_ = get_pmu_value(fdr1_);
    count_cas_read2_ = get_pmu_value(fdr2_);
    count_cas_read3_ = get_pmu_value(fdr3_);
    count_cas_read4_ = get_pmu_value(fdr4_);
    count_cas_read5_ = get_pmu_value(fdr5_);

    count_cas_write0_ = get_pmu_value(fdw_);
    count_cas_write1_ = get_pmu_value(fdw1_);
    count_cas_write2_ = get_pmu_value(fdw2_);
    count_cas_write3_ = get_pmu_value(fdw3_);
    count_cas_write4_ = get_pmu_value(fdw4_);
    count_cas_write5_ = get_pmu_value(fdw5_);

    ioctl(fdr_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdr1_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdr2_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdr3_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdr4_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdr5_, PERF_EVENT_IOC_ENABLE, 0);

    ioctl(fdw_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw1_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw2_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw3_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw4_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fdw5_, PERF_EVENT_IOC_ENABLE, 0);
    return;
  }

  virtual void StopCounting(void) {
    ioctl(fdr_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdr1_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdr2_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdr3_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdr4_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdr5_, PERF_EVENT_IOC_DISABLE, 0);

    ioctl(fdw_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw1_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw2_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw3_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw4_, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fdw5_, PERF_EVENT_IOC_DISABLE, 0);

    long long countr = 0;
    countr += get_pmu_value(fdr_)  - count_cas_read0_;
    countr += get_pmu_value(fdr1_) - count_cas_read1_;
    countr += get_pmu_value(fdr2_) - count_cas_read2_;
    countr += get_pmu_value(fdr3_) - count_cas_read3_;
    countr += get_pmu_value(fdr4_) - count_cas_read4_;
    countr += get_pmu_value(fdr5_) - count_cas_read5_;

    long long countw = 0;
    countw += get_pmu_value(fdw_)  - count_cas_write0_ ;
    countw += get_pmu_value(fdw1_) - count_cas_write1_ ;
    countw += get_pmu_value(fdw2_) - count_cas_write2_ ;
    countw += get_pmu_value(fdw3_) - count_cas_write3_ ;
    countw += get_pmu_value(fdw4_) - count_cas_write4_ ;
    countw += get_pmu_value(fdw5_) - count_cas_write5_ ;
    
    // IMC transfer * linesize is rough memory traffic
    auto data_reads_mib =  (countr)*llc_cache_linesize_/1024/1024;
    auto data_writes_mib =  (countw)*llc_cache_linesize_/1024/1024;

    std::cout << "MemoryTraffic: " << data_reads_mib << " MiB data_reads" << std::endl;
    std::cout << "MemoryTraffic: " << data_writes_mib << " MiB data_writes" << std::endl;
  }

  private:
    long long get_pmu_value(int fd) {
        long long count = 0;
        auto num_reads = read(fd, &count, sizeof(long long));
        if (num_reads == -1) {
          throw "ERROR reading : PMU DRAM counters";
        }
        return count;
    }


  private:
      int fdr1_;
      int fdw1_;
      int fdr2_;
      int fdw2_;
      int fdr3_;
      int fdw3_;
      int fdr4_;
      int fdw4_;
      int fdr5_;
      int fdw5_;
      long long count_cas_read0_;
      long long count_cas_read1_;
      long long count_cas_read2_;
      long long count_cas_read3_;
      long long count_cas_read4_;
      long long count_cas_read5_;
      long long count_cas_write0_;
      long long count_cas_write1_;
      long long count_cas_write2_;
      long long count_cas_write3_;
      long long count_cas_write4_;
      long long count_cas_write5_;
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
