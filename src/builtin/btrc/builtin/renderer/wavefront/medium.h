#pragma once

#include <btrc/builtin/renderer/wavefront/soa.h>
#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace medium_pipeline_detail
{

    struct SOAParams
    {
        PathSOA         path;
        RaySOA          ray;
        BSDFLeSOA       bsdf_le;
        IntersectionSOA inct;

        PathSOA         output_path;
        RaySOA          output_ray;
        BSDFLeSOA       output_bsdf_le;

        ShadowRaySOA shadow_ray;
    };

    CUJ_PROXY_CLASS(
        CSOAParams, SOAParams,
        path, ray, bsdf_le, inct,
        output_path, output_ray, output_bsdf_le,
        shadow_ray);

} // namespace medium_pipeline_detail

/*
    if miss
        return
    resolve medium
    sample scattering event
    if scattered
        generate shadow ray
        generate next ray
        mark inst_id with INST_ID_MEDIUM_MASK
    else
        modify beta and beta_le
        write back resolved medium id
*/
class MediumPipeline : public Uncopyable
{
public:

    using SOAParams = medium_pipeline_detail::SOAParams;
    using CSOAParams = medium_pipeline_detail::CSOAParams;

    MediumPipeline() = default;

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

    MediumPipeline(MediumPipeline &&other) noexcept;

    MediumPipeline &operator=(MediumPipeline &&other) noexcept;

    void swap(MediumPipeline &other) noexcept;

    void sample_scattering(int total_state_count, const SOAParams &soa);

private:

    RC<cuda::Module>                cuda_module_;
    RC<cuda::Buffer<StateCounters>> state_counters_;
};

BTRC_WFPT_END
