#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/builtin/sampler/independent.h>
#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/optix/pipeline.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace shadow_pipeline_detail
{

    struct LaunchParams
    {
        OptixTraversableHandle handle;

        Vec2u                     *pixel_coord;
        Vec4f                     *ray_o_medium_id;
        Vec4f                     *ray_d_t1;
        Spectrum                  *beta_li;
        IndependentSampler::State *sampler_state;
    };

    CUJ_PROXY_CLASS(
        CLaunchParams,
        LaunchParams,
        handle,
        pixel_coord,
        ray_o_medium_id,
        ray_d_t1,
        beta_li,
        sampler_state);

} // namespace shadow_pipeline_detail

class ShadowPipeline : public Uncopyable
{
public:

    struct SOAParams
    {
        Vec2u                     *pixel_coord;
        Vec4f                     *ray_o_medium_id;
        Vec4f                     *ray_d_t1;
        Spectrum                  *beta_li;
        IndependentSampler::State *sampler_state;
    };

    using LaunchParams = shadow_pipeline_detail::LaunchParams;
    using CLaunchParams = shadow_pipeline_detail::CLaunchParams;

    ShadowPipeline() = default;

    ShadowPipeline(
        const Scene         &scene,
        Film                &film,
        OptixDeviceContext   context,
        bool                 motion_blur,
        bool                 triangle_only,
        int                  traversable_depth,
        float                world_diagonal);

    ShadowPipeline(ShadowPipeline &&other) noexcept;

    ShadowPipeline &operator=(ShadowPipeline &&other) noexcept;

    void swap(ShadowPipeline &other) noexcept;

    operator bool() const;

    void test(
        OptixTraversableHandle handle,
        int shadow_ray_count,
        const SOAParams &soa_params) const;

private:

    optix::SimpleOptixPipeline pipeline_;
    mutable cuda::Buffer<LaunchParams> device_launch_params_;
};

BTRC_WFPT_END
