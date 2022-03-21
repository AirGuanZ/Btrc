#pragma once

#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

#include "./volume.h"

BTRC_WFPT_BEGIN

namespace shade_pipeline_detail
{

    struct SOAParams
    {
        CRNG::Data *rng;

        // per path

        Spectrum *path_radiance;
        Vec2u    *pixel_coord;
        int32_t  *depth;
        Spectrum *beta;

        // for computing mis le

        Spectrum *beta_le_bsdf_pdf;

        // last intersection

        uint32_t *path_flag;
        Vec4u    *inct_t_prim_uv;

        // last ray

        Vec4f    *ray_o_medium_id;
        Vec4f    *ray_d_t1;
        Vec2u    *ray_time_mask;

        int32_t *next_state_index;

        CRNG::Data *output_rng;

        // ouput when active

        Spectrum *output_path_radiance;
        Vec2u *output_pixel_coord;
        int32_t  *output_depth;
        Spectrum *output_beta;

        // for shadow ray

        Vec2u    *output_shadow_pixel_coord;
        Vec4f    *output_shadow_ray_o_medium_id;
        Vec4f    *output_shadow_ray_d_t1;
        Vec2u    *output_shadow_ray_time_mask;
        Spectrum *output_shadow_beta_li;

        // for next ray

        Vec4f    *output_new_ray_o_medium_id;
        Vec4f    *output_new_ray_d_t1;
        Vec2u    *output_new_ray_time_mask;

        Spectrum *output_beta_le_bsdf_pdf;
    };

    CUJ_PROXY_CLASS(
        CSOAParams, SOAParams,
        rng,
        path_radiance,
        pixel_coord,
        depth,
        beta,
        beta_le_bsdf_pdf,
        path_flag,
        inct_t_prim_uv,
        ray_o_medium_id,
        ray_d_t1,
        ray_time_mask,
        next_state_index,
        output_rng,
        output_path_radiance,
        output_pixel_coord,
        output_depth,
        output_beta,
        output_shadow_pixel_coord,
        output_shadow_ray_o_medium_id,
        output_shadow_ray_d_t1,
        output_shadow_ray_time_mask,
        output_shadow_beta_li,
        output_new_ray_o_medium_id,
        output_new_ray_d_t1,
        output_new_ray_time_mask,
        output_beta_le_bsdf_pdf);

} // namespace shade_pipeline_detail

class ShadePipeline : public Uncopyable
{
public:

    using SOAParams = shade_pipeline_detail::SOAParams;
    using CSOAParams = shade_pipeline_detail::CSOAParams;

    ShadePipeline() = default;

    void record_device_code(
        CompileContext      &cc,
        Film                &film,
        const Scene         &scene,
        const VolumeManager &vols,
        const ShadeParams   &shade_params,
        float                world_diagonal);

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
        CompileContext      &cc,
        float                world_diagonal,
        const VolumeManager &vols,
        const LightSampler *light_sampler,
        ref<CSOAParams>     soa_params,
        i32                 soa_index,
        ref<CSpectrum>      path_rad,
        ref<CRNG>           rng,
        boolean             scattered);

    RC<cuda::Module>                kernel_;
    const GeometryInfo             *geo_info_ = nullptr;
    const InstanceInfo             *inst_info_ = nullptr;
    RC<cuda::Buffer<StateCounters>> counters_;
};

BTRC_WFPT_END
