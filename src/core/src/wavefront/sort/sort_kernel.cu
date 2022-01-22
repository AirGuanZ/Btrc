#include "sort_kernel.h"

BTRC_WAVEFRONT_BEGIN

namespace
{

    BTRC_KERNEL void sort_kernel(
        const float    *inct_t,
        int32_t         total_state_count,
        int32_t        *output_state_index,
        int32_t        *active_state_counter,
        int32_t        *inactive_state_counter)
    {
        const int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(thread_idx >= total_state_count)
            return;
        const float t = inct_t[thread_idx];
        if(t >= 0)
        {
            const int index = atomicAdd(active_state_counter, 1);
            output_state_index[thread_idx] = index;
        }
        else
        {
            const int reverse_index = atomicAdd(inactive_state_counter, 1);
            const int index = total_state_count - 1 - reverse_index;
            output_state_index[thread_idx] = index;
        }
    }

} // namespace anonymous

BTRC_CPU void sort_states(
    int          total_state_count,
    const float *inct_t,
    int32_t     *output_active_state_index,
    int32_t     *active_state_counter,
    int32_t     *inactive_state_counter)
{
    constexpr int BLOCK_DIM = 256;
    const int block_cnt = up_align(total_state_count, BLOCK_DIM) / BLOCK_DIM;
    sort_kernel<<<block_cnt, BLOCK_DIM>>>(
        inct_t,
        total_state_count,
        output_active_state_index,
        active_state_counter,
        inactive_state_counter);
}

BTRC_WAVEFRONT_END
