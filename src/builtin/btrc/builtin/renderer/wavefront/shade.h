#pragma once

#include <btrc/builtin/renderer/wavefront/soa.h>
#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace shade_pipeline_detail
{

    struct SOAParams
    {
        PathSOA         path;
        RaySOA          ray;
        BSDFLeSOA       bsdf_le;
        IntersectionSOA inct;

        PathSOA   output_path;
        RaySOA    output_ray;
        BSDFLeSOA output_bsdf_le;

        ShadowRaySOA shadow_ray;
    };

    CUJ_PROXY_CLASS(
        CSOAParams, SOAParams,
        path, ray, bsdf_le, inct,
        output_path, output_ray, output_bsdf_le,
        shadow_ray);

} // namespace shade_pipeline_detail

class ShadePipeline : public Uncopyable
{
public:

    using SOAParams = shade_pipeline_detail::SOAParams;
    using CSOAParams = shade_pipeline_detail::CSOAParams;

    ShadePipeline() = default;

    void record_device_code(
        CompileContext    &cc,
        Film              &film,
        const Scene       &scene,
        const ShadeParams &shade_params,
        float              world_diagonal);

    void initialize(
        RC<cuda::Module>                cuda_module,
        RC<cuda::Buffer<StateCounters>> counters,
        const Scene                    &scene);

    ShadePipeline(ShadePipeline &&other) noexcept;

    ShadePipeline &operator=(ShadePipeline &&other) noexcept;

    void swap(ShadePipeline &other) noexcept;

    void shade(int total_state_count, const SOAParams &soa);

private:

    RC<cuda::Module>                kernel_;
    RC<cuda::Buffer<StateCounters>> counters_;
};

BTRC_WFPT_END
