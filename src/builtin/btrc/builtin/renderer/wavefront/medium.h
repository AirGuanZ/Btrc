#pragma once

#include <btrc/builtin/renderer/wavefront/volume.h>
#include <btrc/core/film.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace medium_pipeline_detail
{

    struct SOAParams
    {
        // rng is updated if not scattered

        IndependentSampler::State *sampler_state;

        // per path

        Spectrum *path_radiance;
        Vec2u    *pixel_coord;
        int32_t  *depth;
        Spectrum *beta;

        // beta and beta_le is modified if not scattered

        Spectrum *beta_le_bsdf_pdf;

        // last intersection

        uint32_t *path_flag;
        Vec4u    *inct_t_prim_uv;

        // last ray
        // resolved medium id will always be stored

        Vec4f    *ray_o_medium_id;
        Vec4f    *ray_d_t1;
        Vec2u    *ray_time_mask;

        // output only when scattered
        // and mark original inst_id with INST_ID_MEDIUM_MASK

        int32_t *next_state_index;

        IndependentSampler::State *output_sampler_state;
        Spectrum                  *output_path_radiance;
        Vec2u                     *output_pixel_coord;
        int32_t                   *output_depth;
        Spectrum                  *output_beta;
        
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
        sampler_state,
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
        output_sampler_state,
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
        CompileContext      &cc,
        Film                &film,
        const VolumeManager &vols,
        const Scene         &scene,
        const ShadeParams   &shade_params,
        float                world_diagonal);

    void initialize(
        RC<cuda::Module>                cuda_module,
        RC<cuda::Buffer<StateCounters>> counters,
        const Scene                    &scene);

    MediumPipeline(MediumPipeline &&other) noexcept;

    MediumPipeline &operator=(MediumPipeline &&other) noexcept;

    void swap(MediumPipeline &other) noexcept;

    void sample_scattering(int total_state_count, const SOAParams &soa);

private:

    void sample_light(
        CompileContext &cc,
        const Scene    &scene,
        ref<CVec3f>     scatter_pos,
        Sampler        &sampler,
        f32             time,
        ref<CVec3f>     shadow_d,
        ref<f32>        shadow_t1,
        ref<f32>        shadow_light_pdf,
        ref<CSpectrum>  shadow_li) const;

    RC<cuda::Module>                cuda_module_;
    RC<cuda::Buffer<StateCounters>> state_counters_;
};

BTRC_WFPT_END
