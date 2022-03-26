#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/file.h>
#include <btrc/utils/ptx_cache.h>

#include "./gamma.h"

using namespace btrc;

namespace
{

    const char *KERNEL_GAMMA = "gamma";
    const char *CACHE = "./.btrc_cache/gui_gamma.ptx";

    f32 gamma(f32 x)
    {
        return cstd::pow(x, 1 / 2.2f);
    }

    std::string get_kernel_ptx()
    {
        using namespace cuj;

        const std::string cache_filename = (get_executable_filename().parent_path() / CACHE).string();
        auto cached_ptx = load_kernel_cache(cache_filename);
        if(!cached_ptx.empty())
            return cached_ptx;

        ScopedModule cuj_module;

        kernel(KERNEL_GAMMA, [&](
            ptr<CVec4f> color_buffer,
            i32         width,
            i32         height)
        {
            i32 xi = cuj::cstd::thread_idx_x() + cuj::cstd::block_idx_x() * cuj::cstd::block_dim_x();
            i32 yi = cuj::cstd::thread_idx_y() + cuj::cstd::block_idx_y() * cuj::cstd::block_dim_y();
            $if(xi < width & yi < height)
            {
                var i = yi * width + xi;
                ref color = color_buffer[i];
                color.x = gamma(color.x);
                color.y = gamma(color.y);
                color.z = gamma(color.z);
            };
        });

        PTXGenerator gen;
        gen.set_options(Options{
            .opt_level = OptimizationLevel::O3,
            .fast_math = true,
            .approx_math_func = true
            });
        gen.generate(cuj_module);

        create_kernel_cache(cache_filename, gen.get_ptx());
        return gen.get_ptx();
    }

} // namespace anonymous

Gamma::Gamma()
{
    const std::string ptx = get_kernel_ptx();
    module_.load_ptx_from_memory(ptx.data(), ptx.size());
    module_.link();
}

PostProcessor::ExecutionPolicy Gamma::get_execution_policy() const
{
    return ExecutionPolicy::Always;
}

void Gamma::process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height)
{
    constexpr int BLOCK_SIZE = 16;
    const int block_cnt_x = up_align(width, BLOCK_SIZE) / BLOCK_SIZE;
    const int block_cnt_y = up_align(height, BLOCK_SIZE) / BLOCK_SIZE;

    module_.launch(
        KERNEL_GAMMA,
        { block_cnt_x, block_cnt_y },
        { BLOCK_SIZE, BLOCK_SIZE, 1 },
        color,
        width,
        height);
}
