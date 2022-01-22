#include <btrc/core/wavefront/sort.h>

#include "sort/sort_kernel.h"

BTRC_WAVEFRONT_BEGIN

SortPipeline::SortPipeline()
{
    counters_ = CUDABuffer<int32_t>(2);
}

SortPipeline::SortPipeline(SortPipeline &&other) noexcept
    : SortPipeline()
{
    swap(other);
}

SortPipeline &SortPipeline::operator=(SortPipeline &&other) noexcept
{
    swap(other);
    return *this;
}

SortPipeline::operator bool() const
{
    return true;
}

void SortPipeline::swap(SortPipeline &other) noexcept
{
    std::swap(counters_, other.counters_);
}

int SortPipeline::sort(
    int          current_active_state_count,
    const float *inct_t,
    int32_t     *output_active_state_index)
{
    counters_.clear_bytes(0);

    sort_states(
        current_active_state_count, inct_t,
        output_active_state_index,
        counters_.get(), counters_.get() + 1);
    throw_on_error(cudaStreamSynchronize(nullptr));

    int32_t counters[2] = { 0, 0 };
    counters_.to_cpu(counters);
    assert(counters[0] + counters[1] == current_active_state_count);
    return counters[0];
}

BTRC_WAVEFRONT_END
