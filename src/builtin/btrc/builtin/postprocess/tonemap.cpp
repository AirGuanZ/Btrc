#include <btrc/builtin/postprocess/tonemap.h>
#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/file.h>
#include <btrc/utils/ptx_cache.h>

BTRC_BUILTIN_BEGIN

namespace
{

    const char *KERNEL_TONEMAP = "tonemap";
    const char *CACHE = "./.btrc_cache/tonemap.ptx";

    f32 tonemap(f32 x)
    {
        constexpr float tA = 2.51f;
        constexpr float tB = 0.03f;
        constexpr float tC = 2.43f;
        constexpr float tD = 0.59f;
        constexpr float tE = 0.14f;
        return cstd::saturate((x * (tA * x + tB)) / (x * (tC * x + tD) + tE));
    }

    std::string get_kernel_ptx()
    {
        using namespace cuj;

        const std::string cache_filename = (get_executable_filename().parent_path() / CACHE).string();
        auto cached_ptx = load_kernel_cache(cache_filename);
        if(!cached_ptx.empty())
            return cached_ptx;

        ScopedModule cuj_module;

        kernel(KERNEL_TONEMAP, [&](
            ptr<CVec4f> color_buffer,
            i32         width,
            i32         height,
            f32         exposure)
        {
            i32 xi = cstd::thread_idx_x() + cstd::block_idx_x() * cstd::block_dim_x();
            i32 yi = cstd::thread_idx_y() + cstd::block_idx_y() * cstd::block_dim_y();
            $if(xi < width & yi < height)
            {
                var i = yi * width + xi;
                ref color = color_buffer[i];
                color.x = tonemap(color.x * exposure);
                color.y = tonemap(color.y * exposure);
                color.z = tonemap(color.z * exposure);
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

ACESToneMap::ACESToneMap()
{
    const std::string ptx = get_kernel_ptx();
    module_.load_ptx_from_memory(ptx.data(), ptx.size());
    module_.link();
}

void ACESToneMap::set_exposure(float exposure)
{
    exposure_ = exposure;
}

PostProcessor::ExecutionPolicy ACESToneMap::get_execution_policy() const
{
    return ExecutionPolicy::Always;
}

void ACESToneMap::process(Vec4f *color, Vec4f *albedo, Vec4f *normal, int width, int height)
{
    constexpr int BLOCK_SIZE = 16;
    const int block_cnt_x = up_align(width, BLOCK_SIZE) / BLOCK_SIZE;
    const int block_cnt_y = up_align(height, BLOCK_SIZE) / BLOCK_SIZE;

    module_.launch(
        KERNEL_TONEMAP,
        { block_cnt_x, block_cnt_y },
        { BLOCK_SIZE, BLOCK_SIZE, 1 },
        color,
        width,
        height,
        exposure_);
}

RC<PostProcessor> ACESToneMapCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const float exposure = node->parse_child_or("exposure", 1.0f);
    auto ret = newRC<ACESToneMap>();
    ret->set_exposure(exposure);
    return ret;
}

BTRC_BUILTIN_END
