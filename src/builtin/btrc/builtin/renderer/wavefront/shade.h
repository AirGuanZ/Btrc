#pragma once

#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

#include "./common.h"

BTRC_WFPT_BEGIN

namespace shade_pipeline_detail
{

    struct SOAParams
    {
        RNG::Data *rng;

        // per path

        Spectrum *path_radiance;
        Vec2f    *pixel_coord;
        int32_t  *depth;
        Spectrum *beta;

        // for computing mis le

        Spectrum *beta_le;
        float    *bsdf_pdf;

        // last intersection

        Vec2u *inct_inst_launch_index;
        Vec4u *inct_t_prim_uv;

        // last ray

        Vec4f    *ray_o_t0;
        Vec4f    *ray_d_t1;
        Vec2u    *ray_time_mask;
        uint32_t *ray_medium_id;

        // always output

        RNG::Data *output_rng;

        // ouput when active

        Spectrum *output_path_radiance;
        Vec2f    *output_pixel_coord;
        int32_t  *output_depth;
        Spectrum *output_beta;

        // for shadow ray

        Vec2f    *output_shadow_pixel_coord;
        Vec4f    *output_shadow_ray_o_t0;
        Vec4f    *output_shadow_ray_d_t1;
        Vec2u    *output_shadow_ray_time_mask;
        Spectrum *output_shadow_beta_li;
        uint32_t *output_shadow_ray_medium_id;

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
        bsdf_pdf,
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
        output_shadow_ray_medium_id,
        output_new_ray_o_t0,
        output_new_ray_d_t1,
        output_new_ray_time_mask,
        output_new_ray_medium_id,
        output_beta_le,
        output_bsdf_pdf);

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
        const ShadeParams &shade_params);

    void initialize(
        RC<cuda::Module>                cuda_module,
        RC<cuda::Buffer<StateCounters>> counters,
        const Scene                    &scene);

    ShadePipeline(ShadePipeline &&other) noexcept;

    ShadePipeline &operator=(ShadePipeline &&other) noexcept;

    void swap(ShadePipeline &other) noexcept;

    void shade(int total_state_count, const SOAParams &soa);

private:

    void handle_miss(
        CompileContext     &cc,
        const LightSampler *light_sampler,
        ref<CSOAParams>     soa_params,
        u32                 soa_index,
        ref<CSpectrum>      path_rad);

    RC<cuda::Module>                kernel_;
    const GeometryInfo             *geo_info_ = nullptr;
    const InstanceInfo             *inst_info_ = nullptr;
    RC<cuda::Buffer<StateCounters>> counters_;
};

BTRC_WFPT_END
