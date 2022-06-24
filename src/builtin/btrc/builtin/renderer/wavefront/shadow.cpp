#include <btrc/utils/optix/device_funcs.h>

#include "./shadow.h"

BTRC_WFPT_BEGIN

namespace
{

    const char *LAUNCH_PARAMS_NAME     = "launch_params";
    const char *RAYGEN_SHADOW_NAME     = "__raygen__shadow";
    const char *MISS_SHADOW_NAME       = "__miss__shadow";
    const char *CLOSESTHIT_SHADOW_NAME = "__closesthit__shadow";

    std::string generate_shadow_kernel(CompileContext &cc, const Scene &scene, Film &film, float world_diagonal)
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
            
            auto [ray, medium_id] = launch_params.shadow_ray.load_ray(i32(launch_idx));

            optix::trace(
                launch_params.handle,
                ray.o, ray.d, 0, ray.t, 0, u32(optix::RAY_MASK_ALL),
                OPTIX_RAY_FLAG_NONE, 0, 1, 0, launch_idx, medium_id);
        });

        kernel(
            MISS_SHADOW_NAME,
            [&cc, &scene, &film, world_diagonal, global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_payload(0);

            auto [pixel_coord, beta] = launch_params.shadow_ray.load_beta(i32(launch_idx));

            IndependentSampler sampler({ film.width(), film.height() }, launch_params.sampler_state[launch_idx]);

            var ray_o = optix::get_ray_o();
            var ray_d = optix::get_ray_d();
            var ray_t1 = optix::get_ray_tmax();

            var tr = CSpectrum::one();
            CMediumID medium_id = optix::get_payload(1);

            $if(ray_t1 > 1)
            {
                tr = scene.get_volume_primitive_medium()->tr(
                    cc, ray_o, ray_o + normalize(ray_d) * world_diagonal, sampler);
            }
            $else
            {
                var end_pnt = ray_o + ray_d * ray_t1;
                scene.access_medium(i32(medium_id), [&](const Medium *medium)
                {
                    tr = medium->tr(cc, ray_o, end_pnt, ray_o, end_pnt, sampler);
                });
            };
            beta = beta * tr;

            sampler.save(launch_params.sampler_state + launch_idx);
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
    const Scene         &scene,
    Film                &film,
    OptixDeviceContext   context,
    bool                 motion_blur,
    bool                 triangle_only,
    int                  traversable_depth,
    float                world_diagonal)
{
    CompileContext cc;

    pipeline_ = optix::SimpleOptixPipeline(
        context,
        optix::SimpleOptixPipeline::Program{
            .ptx                = generate_shadow_kernel(cc, scene, film, world_diagonal),
            .launch_params_name = LAUNCH_PARAMS_NAME,
            .raygen_name        = RAYGEN_SHADOW_NAME,
            .miss_name          = MISS_SHADOW_NAME,
            .closesthit_name    = CLOSESTHIT_SHADOW_NAME
        },
        optix::SimpleOptixPipeline::Config{
            .payload_count     = 2,
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
        .handle     = handle,
        .shadow_ray = soa_params.shadow_ray,
        .sampler_state = soa_params.sampler_state
    };
    device_launch_params_.from_cpu(&launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr,
        device_launch_params_, sizeof(LaunchParams),
        &pipeline_.get_sbt(), shadow_ray_count, 1, 1));
}

BTRC_WFPT_END
