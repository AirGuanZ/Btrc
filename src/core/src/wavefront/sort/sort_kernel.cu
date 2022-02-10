#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/async/sort.h>

#include "sort_kernel.h"


BTRC_WAVEFRONT_BEGIN

namespace
{

    BTRC_KERNEL void fill_index(int32_t *output_indices, int32_t total_state_count)
    {
        const int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(thread_idx >= total_state_count)
            return;
        output_indices[thread_idx] = thread_idx;
    }

    struct StateCmpLess
    {
        const float *inct_t;
        const Vec4u *inct_uv_id;

        BTRC_GPU bool operator()(int32_t a, int32_t b) const
        {
            const bool a_has_inct = inct_t[a] >= 0;
            const bool b_has_inct = inct_t[b] >= 0;
            if(a_has_inct && !b_has_inct)
                return true;
            if(!a_has_inct && b_has_inct)
                return false;
            const Vec4u a_uv_id = inct_uv_id[a];
            const Vec4u b_uv_id = inct_uv_id[b];
            if(a_uv_id.w < b_uv_id.w)
                return true;
            if(a_uv_id.w > b_uv_id.w)
                return false;
            return a_uv_id.z < b_uv_id.z;
        }
    };

} // namespace anonymous

BTRC_CPU void sort_states(
    int          total_state_count,
    const float *inct_t,
    const Vec4u *inct_uv_id,
    int32_t     *output_active_state_index)
{
    // TODO: segmented sort

    //constexpr int BLOCK_DIM = 256;
    //const int block_cnt = up_align(total_state_count, BLOCK_DIM) / BLOCK_DIM;
    //fill_index<<<block_cnt, BLOCK_DIM>>>(output_active_state_index, total_state_count);
    
    StateCmpLess cmp;
    cmp.inct_t = inct_t;
    cmp.inct_uv_id = inct_uv_id;
    
    auto beg = thrust::device_pointer_cast(output_active_state_index);
    auto end = thrust::device_pointer_cast(output_active_state_index + total_state_count);
    sort(beg, end, cmp);
}

BTRC_WAVEFRONT_END
