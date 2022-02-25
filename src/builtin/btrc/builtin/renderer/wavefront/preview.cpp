#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/file.h>
#include <btrc/utils/ptx_cache.h>

#include "./preview.h"

BTRC_WFPT_BEGIN

namespace
{

    const char *KERNEL = "generate_preview_image";
    const char *CACHE  = "./.btrc_cache/wfpt_preview.ptx";

    std::string generate_kernel_ptx()
    {
        using namespace cuj;

        const std::string cache_filename = (get_executable_filename().parent_path() / CACHE).string();

        auto cached_ptx = load_kernel_cache(cache_filename);
        if(!cached_ptx.empty())
            return cached_ptx;

        ScopedModule cuj_module;

        kernel(KERNEL, [&](
            i32         width,
            i32         height,
            ptr<CVec4f> value_buffer,
            ptr<f32>    weight_buffer,
            ptr<CVec4f> output_buffer)
        {
            i32 xi = cstd::thread_idx_x() + cstd::block_idx_x() * cstd::block_dim_x();
            i32 yi = cstd::thread_idx_y() + cstd::block_idx_y() * cstd::block_dim_y();
            $if(xi < width & yi < height)
            {
                i32 i = yi * width + xi;
                CVec4f value = load_aligned(value_buffer + i);
                f32 weight = weight_buffer[i];
                CVec3f output;
                $if(weight > 0)
                {
                    output = value.xyz() / weight;
                };
                output.x = cstd::pow(output.x, 1 / 2.2f);
                output.y = cstd::pow(output.y, 1 / 2.2f);
                output.z = cstd::pow(output.z, 1 / 2.2f);
                save_aligned(CVec4f(output, 1), output_buffer + i);
            };
        });

        PTXGenerator gen;
        gen.set_options(Options{
            .opt_level        = OptimizationLevel::O3,
            .fast_math        = true,
            .approx_math_func = true
        });
        gen.generate(cuj_module);
        
        create_kernel_cache(cache_filename, gen.get_ptx());
        return gen.get_ptx();
    }

} // namespace anonymous

PreviewImageGenerator::PreviewImageGenerator()
{
    const auto ptx = generate_kernel_ptx();
    cuda_module_.load_ptx_from_memory(ptx.data(), ptx.size());
    cuda_module_.link();
}

void PreviewImageGenerator::generate(
    int          width,
    int          height,
    const Vec4f *value_buffer,
    const float *weight_buffer,
    Vec4f       *output_buffer) const
{
    constexpr int BLOCK_SIZE = 16;
    const int block_cnt_x = up_align(width, BLOCK_SIZE) / BLOCK_SIZE;
    const int block_cnt_y = up_align(height, BLOCK_SIZE) / BLOCK_SIZE;

    cuda_module_.launch(
        KERNEL,
        { block_cnt_x, block_cnt_y, 1 },
        { BLOCK_SIZE, BLOCK_SIZE, 1 },
        width,
        height,
        value_buffer,
        weight_buffer,
        output_buffer);
}

BTRC_WFPT_END
