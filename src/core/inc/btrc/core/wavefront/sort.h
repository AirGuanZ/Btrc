#pragma once

#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/cuda/module.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_WAVEFRONT_BEGIN

class SortPipeline : public Uncopyable
{
public:

    SortPipeline();

    SortPipeline(SortPipeline &&other) noexcept;

    SortPipeline &operator=(SortPipeline &&other) noexcept;

    operator bool() const;

    void swap(SortPipeline &other) noexcept;

    void sort(
        int          current_active_state_count,
        const float *inct_t,
        int32_t     *output_active_state_index);

private:

    CUDABuffer<int32_t> counters_;
};

BTRC_WAVEFRONT_END
