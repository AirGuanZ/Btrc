#include <vector_types.h>

#include "./wfpt_preview.h"

BTRC_WFPT_BEGIN

namespace
{

    BTRC_KERNEL void compute_preview_image_kernel(
        int          width,
        int          height,
        const Vec4f *value_buffer,
        const float *weight_buffer,
        Vec4f       *output_buffer)
    {
        const int xi = threadIdx.x + blockIdx.x * blockDim.x;
        const int yi = threadIdx.y + blockIdx.y * blockDim.y;
        if(xi < width && yi < height)
        {
            const int i = yi * width + xi;
            const Vec4f value = value_buffer[i];
            const float weight = weight_buffer[i];
            const Vec3f output = weight > 0 ? value.xyz() / weight : Vec3f(0);
            output_buffer[i] = Vec4f(
                btrc_pow(output.x, 1 / 2.2f),
                btrc_pow(output.y, 1 / 2.2f),
                btrc_pow(output.z, 1 / 2.2f), 1.0f);
        }
    }

} // namespace anonymous

void compute_preview_image(
    int          width,
    int          height,
    const Vec4f *value_buffer,
    const float *weight_buffer,
    Vec4f       *output_buffer)
{
    constexpr int BLOCK_SIZE = 16;
    const int block_cnt_x = up_align(width, BLOCK_SIZE) / BLOCK_SIZE;
    const int block_cnt_y = up_align(height, BLOCK_SIZE) / BLOCK_SIZE;
    const dim3 block_cnt(block_cnt_x, block_cnt_y);
    const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    compute_preview_image_kernel<<<block_cnt, block_size>>>(
        width, height, value_buffer, weight_buffer, output_buffer);
}

BTRC_WFPT_END
