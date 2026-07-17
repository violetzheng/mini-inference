#include "tensor/parallel_for.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace mini_inference::tensor
{

    namespace
    {

        // Below this many items per thread, spawning threads costs more than it saves.
        constexpr std::size_t kMinItemsPerThread = 64;

    } // namespace

    void parallel_for(std::size_t begin, std::size_t end, const std::function<void(std::size_t, std::size_t)> &body)
    {
        const std::size_t total = end - begin;
        const unsigned int hw_threads = std::thread::hardware_concurrency();
        const std::size_t max_threads = hw_threads == 0 ? 1 : static_cast<std::size_t>(hw_threads);
        const std::size_t num_threads = std::min(max_threads, total / kMinItemsPerThread);

        if (num_threads <= 1)
        {
            body(begin, end);
            return;
        }

        const std::size_t chunk_size = (total + num_threads - 1) / num_threads;

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            const std::size_t chunk_begin = begin + t * chunk_size;
            const std::size_t chunk_end = std::min(end, chunk_begin + chunk_size);
            if (chunk_begin >= chunk_end)
            {
                break;
            }
            workers.emplace_back(body, chunk_begin, chunk_end);
        }

        for (std::thread &worker : workers)
        {
            worker.join();
        }
    }

} // namespace mini_inference::tensor
