#include <thread>
#include <vector>
#include <cmath>

class ThreadPool {
 
public:
 
	template<typename Index, typename Callable>
	static void ParallelFor(Index start, Index end, Callable func, unsigned int num_threads) {
		// Estimate number of threads in the pool
		const static unsigned nb_threads = num_threads;
 
		// Size of a slice for the range functions
		Index n = end - start + 1;
		Index slice = (Index) std::round(n / static_cast<double> (nb_threads));
		slice = std::max(slice, Index(1));
 
		// [Helper] Inner loop
		auto launchRange = [&func] (unsigned int k1, unsigned int k2) {
			for (Index k = k1; k < k2; k++) {
				func(k);
			}
		};
 
		// Create pool and launch jobs
		std::vector<std::thread> pool;
		pool.reserve(nb_threads);
		Index i1 = start;
		Index i2 = std::min(start + slice, end);
		for (unsigned i = 0; i + 1 < nb_threads && i1 < end; ++i) {
			pool.emplace_back(launchRange, i1, i2);
			i1 = i2;
			i2 = std::min(i2 + slice, end);
		}
		if (i1 < end) {
			pool.emplace_back(launchRange, i1, end);
		}
 
		// Wait for jobs to finish
		for (std::thread &t : pool) {
			if (t.joinable()) {
				t.join();
			}
		}
	}
 
	// Serial version for easy comparison
	template<typename Index, typename Callable>
	static void SequentialFor(Index start, Index end, Callable func) {
		for (Index i = start; i < end; i++) {
			func(i);
		}
	}
 
};
