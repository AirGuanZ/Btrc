#include <btrc/utils/optix/device_funcs.h>

#include "./shadow.h"

BTRC_WFPT_BEGIN

namespace
{

    const char *LAUNCH_PARAMS_NAME     = "launch_params";
    const char *RAYGEN_SHADOW_NAME     = "__raygen__shadow";
    const char *MISS_SHADOW_NAME       = "__miss__shadow";
    const char *CLOSESTHIT_SHADOW_NAME = "__closesthit__shadow";

    std::string generate_shadow_kernel(CompileContext &cc, const Scene &scene, Film &film)
    {
        using namespace cuj;

        ScopedModule cuj_module;

        auto global_launch_params = allocate_constant_memory<
            ShadowPipeline::CLaunchParams>(LAUNCH_PARAMS_NAME);

        kernel(
            RAYGEN_SHADOW_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_launch_index_x();

            CUJ_ASSERT(bitcast<u64>(launch_params.ray_d_t1) != 0);

            var o_t0 = load_aligned(launch_params.ray_o_t0 + launch_idx);
            var d_t1 = load_aligned(launch_params.ray_d_t1 + launch_idx);
            var time_mask = load_aligned(launch_params.ray_time_mask + launch_idx);

            var o = o_t0.xyz();
            var d = d_t1.xyz();
            var t0 = o_t0.w;
            var t1 = d_t1.w;
            var time = bitcast<f32>(time_mask.x);
            var mask = time_mask.y;

            optix::trace(
                launch_params.handle,
                o, d, t0, t1, time, mask,
                OPTIX_RAY_FLAG_NONE, 0, 1, 0, launch_idx);
        });

        kernel(
            MISS_SHADOW_NAME,
            [&cc, &scene, &film, global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_payload(0);

            var pixel_coord = load_aligned(launch_params.pixel_coord + launch_idx);
            var beta = load_aligned(launch_params.beta_li + launch_idx);

            var medium_id = launch_params.ray_medium_id[launch_idx];
            $if(medium_id != MEDIUM_ID_VOID)
            {
                var tr = CSpectrum::one();
                var ray_o = optix::get_ray_o();
                var ray_d = optix::get_ray_d();
                var ray_t1 = optix::get_ray_tmax();
                $switch(medium_id)
                {
                    for(int i = 0; i < scene.get_medium_count(); ++i)
                    {
                        $case(i)
                        {
                            tr = scene.get_medium(i)->tr(cc, ray_o, ray_o + ray_d * ray_t1);
                        };
                    }
                    $default
                    {
                        cstd::unreachable();
                    };
                };
                beta = beta * tr;
            };

            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, beta.to_rgb());
        });

        kernel(CLOSESTHIT_SHADOW_NAME, [] { });

        PTXGenerator gen;
        gen.set_options(Options{
            .opt_level        = OptimizationLevel::O3,
            .fast_math        = true,
            .approx_math_func = true
        });
        gen.generate(cuj_module);

        return gen.get_ptx();
    }

} // namespace anonymous

ShadowPipeline::ShadowPipeline(
    bool               offline_mode,
    const Scene       &scene,
    Film              &film,
    OptixDeviceContext context,
    bool               motion_blur,
    bool               triangle_only,
    int                traversable_depth)
{
    CompileContext cc(offline_mode);

    pipeline_ = optix::SimpleOptixPipeline(
        context,
        optix::SimpleOptixPipeline::Program{
            .ptx                = generate_shadow_kernel(cc, scene, film),
            .launch_params_name = LAUNCH_PARAMS_NAME,
            .raygen_name        = RAYGEN_SHADOW_NAME,
            .miss_name          = MISS_SHADOW_NAME,
            .closesthit_name    = CLOSESTHIT_SHADOW_NAME
        },
        optix::SimpleOptixPipeline::Config{
            .payload_count     = 1,
            .traversable_depth = traversable_depth,
            .motion_blur       = motion_blur,
            .triangle_only     = triangle_only
        });
    device_launch_params_ = cuda::Buffer<LaunchParams>(1);
}

ShadowPipeline::ShadowPipeline(ShadowPipeline &&other) noexcept
    : ShadowPipeline()
{
    swap(other);
}

ShadowPipeline &ShadowPipeline::operator=(ShadowPipeline &&other) noexcept
{
    swap(other);
    return *this;
}

void ShadowPipeline::swap(ShadowPipeline &other) noexcept
{
    pipeline_.swap(other.pipeline_);
    device_launch_params_.swap(other.device_launch_params_);
}

ShadowPipeline::operator bool() const
{
    return pipeline_;
}

void ShadowPipeline::test(
    OptixTraversableHandle handle,
    int shadow_ray_count,
    const SOAParams &soa_params) const
{
    const LaunchParams launch_params = {
        .handle        = handle,
        .pixel_coord   = soa_params.pixel_coord,
        .ray_o_t0      = soa_params.ray_o_t0,
        .ray_d_t1      = soa_params.ray_d_t1,
        .ray_time_mask = soa_params.ray_time_mask,
        .ray_medium_id = soa_params.ray_medium_id,
        .beta_li       = soa_params.beta_li
    };
    device_launch_params_.from_cpu(&launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr,
        device_launch_params_, sizeof(LaunchParams),
        &pipeline_.get_sbt(), shadow_ray_count, 1, 1));
}

BTRC_WFPT_END
