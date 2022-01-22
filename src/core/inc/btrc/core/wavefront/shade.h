#pragma once

#include <btrc/core/film/film.h>
#include <btrc/core/utils/cuda/module.h>
#include <btrc/core/utils/uncopyable.h>
#include <btrc/core/wavefront/scene.h>
#include <btrc/core/wavefront/soa.h>

/*
r = generate ray
beta = 1
beta_di = 1
loop
{
    inct = trace r
    if !inct
        result += env * beta_di
        exit
    sample light
    eval bsdf
    generate shadow ray
    r = sample bsdf
    update beta_di
    update beta
}
*/

BTRC_WAVEFRONT_BEGIN

namespace shade_pipeline_detail
{

    struct SOAParams
    {
        cstd::LCGData *rng;

        int32_t *active_state_indices;

        // per path

        float   *path_radiance;
        Vec2f   *pixel_coord;
        int32_t *depth;
        float   *beta;

        // for computing mis le

        float *beta_le;
        float *bsdf_pdf;

        // last intersection

        float *inct_t;
        Vec4u *inct_uv_id;

        // last ray

        Vec4f *ray_o_t0;
        Vec4f *ray_d_t1;
        Vec2u *ray_time_mask;

        // always output

        cstd::LCGData *output_rng;

        // ouput when active

        float   *output_path_radiance;
        Vec2f   *output_pixel_coord;
        int32_t *output_depth;
        float   *output_beta;

        // for shadow ray

        Vec2f *output_shadow_pixel_coord;
        Vec4f *output_shadow_ray_o_t0;
        Vec4f *output_shadow_ray_d_t1;
        Vec2u *output_shadow_ray_time_mask;
        float *output_shadow_beta_li;

        // for next ray

        Vec4f *output_new_ray_o_t0;
        Vec4f *output_new_ray_d_t1;
        Vec2u *output_new_ray_time_mask;

        float *output_beta_le;
        float *output_bsdf_pdf;
    };

    CUJ_PROXY_CLASS(
        CSOAParams, SOAParams,
        rng,
        active_state_indices,
        path_radiance,
        pixel_coord,
        depth,
        beta,
        beta_le,
        bsdf_pdf,
        inct_t,
        inct_uv_id,
        ray_o_t0,
        ray_d_t1,
        ray_time_mask,
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
        output_new_ray_o_t0,
        output_new_ray_d_t1,
        output_new_ray_time_mask,
        output_beta_le,
        output_bsdf_pdf);

} // namespace shade_pipeline_detail

class ShadePipeline : public Uncopyable
{
public:

    using SOAParams = shade_pipeline_detail::SOAParams;
    using CSOAParams = shade_pipeline_detail::CSOAParams;

    struct TracingParams
    {
        int   min_depth    = 4;
        int   max_depth    = 8;
        float rr_threshold = 0.2f;
        float rr_cont_prob = 0.6f;
    };

    struct StateCounters
    {
        int active_state_counter = 0;
        int shadow_ray_counter = 0;
    };

    ShadePipeline() = default;

    ShadePipeline(
        Film                &film,
        const SceneData     &scene,
        const SpectrumType  *spectrum_type,
        const TracingParams &tracing_params);

    ShadePipeline(ShadePipeline &&other) noexcept;

    ShadePipeline &operator=(ShadePipeline &&other) noexcept;

    void swap(ShadePipeline &other) noexcept;

    operator bool() const;

    // returns active state count
    StateCounters shade(
        int total_state_count,
        const SOAParams &soa);

private:

    void initialize(
        Film                &film,
        const SceneData     &scene,
        const SpectrumType  *spectrum_type,
        const TracingParams &tracing_params);

    void handle_miss(
        ref<CSOAParams> soa_params, i32 soa_index, CSpectrum &path_rad);

    CUDAModule          kernel_;
    const SceneData    *scene_ = nullptr;
    const SpectrumType *spectrum_type_;

    CUDABuffer<int32_t>      counters_;
    CUDABuffer<int32_t>      inst_id_to_mat_id_;
    CUDABuffer<InstanceInfo> instances_;

    TracingParams tracing_params_;
};

BTRC_WAVEFRONT_END
