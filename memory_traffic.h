#ifdef MEMORY_TRAFFIC_COUNT
#ifndef _MEMORY_TRAFFIC
#define _MEMORY_TRAFFIC

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <asm/unistd.h>



class MemoryTraffic 
{
 public:
  MemoryTraffic(bool clear_cache = true) : fd_(-1), llc_cache_linesize_(0) {
    memset(&pe_, 0, sizeof(struct perf_event_attr));
    pe_.type = PERF_TYPE_HARDWARE;
    pe_.config = PERF_COUNT_HW_CACHE_MISSES; 
    pe_.disabled = 1;
    pe_.exclude_kernel = 1; // exclude events taking place in kernel
    pe_.exclude_hv = 1;     // Exclude hypervisor

    // Get LLC Cacheline size
    get_cacheline_size();

    // If we are to do cold caches then we overwrite existing
    // caches to make sure no kernel used data is in cache
    if (clear_cache)
        overwrite_caches();

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

  inline MemoryTraffic& StartCounting(void) {
    ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    return *this;
  }
  
  inline long long StopCounting(void) {
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    long long count;
    read(fd_, &count, sizeof(long long));
    // LLC miss * linesize is rough memory traffic
    return count*llc_cache_linesize_;
  }

 protected:
  long perf_event_open(struct perf_event_attr*
    hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
  {
    int ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
     group_fd, flags);

    return ret;
  }

 void overwrite_caches() {

   // Get 512 MB for source and copy it to 512 MB dst. 
   // Intention is to copy more memory than it can be fead into cache 
   size_t size_of_floats = 128*1024*1024;
   float *buf;
   int ret = posix_memalign((void**)&buf,
           llc_cache_linesize_,size_of_floats*sizeof(float));
   if (ret != 0) {
     std::cout << "Allocation error of source buffer!" << std::endl;
     exit(-1);
   }

   // Generate some random data 
   for(unsigned int i=0; i < size_of_floats; ++i) {
     buf[i] = i;
   } 
   
   free(buf);
 }
 
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


 private:
  int fd_;
  size_t llc_cache_linesize_;
  struct perf_event_attr pe_;
};

#endif
#endif
