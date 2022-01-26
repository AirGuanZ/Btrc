#include <btrc/core/wavefront/shadow.h>

BTRC_WAVEFRONT_BEGIN

namespace
{

    const char *LAUNCH_PARAMS_NAME     = "launch_params";
    const char *RAYGEN_SHADOW_NAME     = "__raygen__shadow";
    const char *MISS_SHADOW_NAME       = "__miss__shadow";
    const char *CLOSESTHIT_SHADOW_NAME = "__closesthit__shadow";

    std::string generate_shadow_kernel(
        Film &film, const SpectrumType *spec_type)
    {
        using namespace cuj;

        ScopedModule cuj_module;

        auto global_launch_params = allocate_constant_memory<
            ShadowPipeline::CLaunchParams>(LAUNCH_PARAMS_NAME);

        kernel(
            RAYGEN_SHADOW_NAME,
            [&film, spec_type, global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();

        });

        // TODO
        return {};
    }

} // namespace anonymous

ShadowPipeline::ShadowPipeline(
    Film               &film,
    const SpectrumType *spec_type,
    OptixDeviceContext  context,
    bool                motion_blur,
    bool                triangle_only,
    int                 traversable_depth)
{
    pipeline_ = optix::SimpleOptixPipeline(
        context,
        optix::SimpleOptixPipeline::Program{
            .ptx                = generate_shadow_kernel(film, spec_type),
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

    device_launch_params_ = CUDABuffer<LaunchParams>(1);
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
    int                 shadow_ray_count,
    const LaunchParams &launch_params) const
{
    device_launch_params_.from_cpu(&launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr,
        device_launch_params_, sizeof(LaunchParams),
        &pipeline_.get_sbt(), shadow_ray_count, 1, 1));
}

BTRC_WAVEFRONT_END
