#pragma once

#include <btrc/core/film/film.h>
#include <btrc/core/spectrum/spectrum.h>
#include <btrc/core/utils/optix/pipeline.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_WAVEFRONT_BEGIN

namespace shadow_pipeline_detail
{

    struct LaunchParams
    {
        OptixTraversableHandle handle;
        Vec2f *pixel_coord;
        Vec4f *ray_o_t0;
        Vec4f *ray_d_t1;
        Vec2u *ray_time_mask;
        float *beta_li;
    };

    CUJ_PROXY_CLASS(
        CLaunchParams, LaunchParams,
        pixel_coord, ray_o_t0, ray_d_t1, ray_time_mask, beta_li);

} // namespace shadow_pipeline_detail

class ShadowPipeline : public Uncopyable
{
public:

    using LaunchParams = shadow_pipeline_detail::LaunchParams;
    using CLaunchParams = shadow_pipeline_detail::CLaunchParams;

    ShadowPipeline() = default;

    ShadowPipeline(
        Film               &film,
        const SpectrumType *spec_type,
        OptixDeviceContext  context,
        bool                motion_blur,
        bool                triangle_only,
        int                 traversable_depth);

    ShadowPipeline(ShadowPipeline &&other) noexcept;

    ShadowPipeline &operator=(ShadowPipeline &&other) noexcept;

    void swap(ShadowPipeline &other) noexcept;

    operator bool() const;

    void test(
        int                 shadow_ray_count,
        const LaunchParams &launch_params) const;

private:

    optix::SimpleOptixPipeline pipeline_;
    mutable CUDABuffer<LaunchParams> device_launch_params_;
};

BTRC_WAVEFRONT_END
