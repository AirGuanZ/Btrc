#pragma once

#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

#include "./common.h"

BTRC_WFPT_BEGIN

namespace medium_pipeline_detail
{

    struct SOAParams
    {
        // rng is updated if not scattered

        CRNG::Data *rng;

        // per path

        Spectrum *path_radiance;
        Vec2f    *pixel_coord;
        int32_t  *depth;
        Spectrum *beta;

        // beta and beta_le is modified if not scattered

        Spectrum *beta_le;

        // last intersection

        Vec2u *inct_inst_launch_index;
        Vec4u *inct_t_prim_uv;

        // last ray
        // resolved medium id will always be stored

        Vec4f    *ray_o_t0;
        Vec4f    *ray_d_t1;
        Vec2u    *ray_time_mask;
        uint32_t *ray_medium_id;

        // output only when scattered
        // and mark original inst_id with INST_ID_MEDIUM_MASK

        CRNG::Data *output_rng;
        Spectrum  *output_path_radiance;
        Vec2f     *output_pixel_coord;
        int32_t   *output_depth;
        Spectrum  *output_beta;
        
        Vec2f    *output_shadow_pixel_coord;
        Vec4f    *output_shadow_ray_o_t0;
        Vec4f    *output_shadow_ray_d_t1;
        Vec2u    *output_shadow_ray_time_mask;
        Spectrum *output_shadow_beta_li;
        uint32_t *output_shadow_medium_id;

        // for next ray

        Vec4f    *output_new_ray_o_t0;
        Vec4f    *output_new_ray_d_t1;
        Vec2u    *output_new_ray_time_mask;
        uint32_t *output_new_ray_medium_id;

        Spectrum *output_beta_le;
        float    *output_bsdf_pdf;
    };

    CUJ_PROXY_CLASS(
        CSOAParams, SOAParams,
        rng,
        path_radiance,
        pixel_coord,
        depth,
        beta,
        beta_le,
        inct_inst_launch_index,
        inct_t_prim_uv,
        ray_o_t0,
        ray_d_t1,
        ray_time_mask,
        ray_medium_id,
        output_rng,
        output_path_radiance,
        output_pixel_coord,
        output_depth,
        output_beta,
        output_shadow_pixel_coord,
        output_shadow_ray_o_t0,
        output_shadow_ray_d_t1,
        output_shadow_ray_time_mask,
        output_shadow_beta_li,
        output_shadow_medium_id,
        output_new_ray_o_t0,
        output_new_ray_d_t1,
        output_new_ray_time_mask,
        output_new_ray_medium_id,
        output_beta_le,
        output_bsdf_pdf);

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
        const ShadeParams &shade_params);

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
    const GeometryInfo             *geo_info_ = nullptr;
    const InstanceInfo             *inst_info_ = nullptr;
};

BTRC_WFPT_END
