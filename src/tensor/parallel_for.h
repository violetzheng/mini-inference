#pragma once

#include <cstddef>
#include <functional>

namespace mini_inference::tensor
{

    // Splits [begin, end) into contiguous chunks and runs body(chunk_begin, chunk_end)
    // across up to hardware_concurrency() threads, joining before returning. Falls back
    // to a single inline call to body(begin, end) when the range is too small for
    // threading to pay off (thread-spawn overhead would dominate).
    void parallel_for(std::size_t begin, std::size_t end,
                       const std::function<void(std::size_t, std::size_t)> &body);

} // namespace mini_inference::tensor
