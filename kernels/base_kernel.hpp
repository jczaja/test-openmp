#ifndef _BASEKERNEL
#define _BASEKERNEL

#include <string>
#include <toolbox.h>

#define COMMA ,
#define REGISTER_KERNEL(T) extern std::unordered_map<std::string, BaseKernel*> kernels; \
                            static T objectT;

#define REGISTER_KERNEL_VARIANT(T,suf) static T objectT##suf

class BaseKernel
{
  public: 
    virtual void Init(platform_info &pi, int n, int c, int h, int w) = 0;
    virtual void ShowInfo(bool cold_caches) = 0;
    virtual void RunSingle(void) = 0;
    void Run(int num_reps, bool is_xeon, bool cold_caches) {
      // Warming up caches
      for(int n = 0; n < 1; ++n) {
        RunSingle();
      }
#ifdef MEMORY_TRAFFIC_COUNT
      MemoryTraffic* mt = is_xeon ? new XeonMemoryTraffic(cold_caches) : new MemoryTraffic(cold_caches);
#endif
#ifdef RUNTIME_TEST
      auto rt = Runtime(tsc_ghz_,cold_caches);
#endif
#ifdef MEMORY_TRAFFIC_COUNT
        mt->StartCounting();
#endif
      for(int n = 0; n< num_reps; ++n) {
#ifdef RUNTIME_TEST
        rt.Start();
#endif
        RunSingle();  // Single iteration execution
            //sleep(1);
#ifdef RUNTIME_TEST
        rt.Stop();
#endif
      }
#ifdef MEMORY_TRAFFIC_COUNT
        mt->StopCounting();
        delete mt;
#endif
    }

  public:
   unsigned long long tsc_ghz_;
};


#endif
