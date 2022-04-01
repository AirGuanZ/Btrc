#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/core/camera.h>
#include <btrc/core/film.h>
#include <btrc/core/film_filter.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace generate_pipeline_detail
{

    struct SOAParams
    {
        GlobalSampler::State *output_sampler_state;
        Vec2u                *output_pixel_coord;
        Vec4f                *output_ray_o_medium_id;
        Vec4f                *output_ray_d_t1;
        Spectrum             *output_beta;
        Spectrum             *output_beta_le_bsdf_pdf;
        int                  *output_depth;
        Spectrum             *output_path_radiance;
    };

    CUJ_PROXY_CLASS(
        CSOAParams,
        SOAParams,
        output_sampler_state,
        output_pixel_coord,
        output_ray_o_medium_id,
        output_ray_d_t1,
        output_beta,
        output_beta_le_bsdf_pdf,
        output_depth,
        output_path_radiance);

} // namespace generate_pipeline_detail

class GeneratePipeline : public Uncopyable
{
public:

    using SOAParams = generate_pipeline_detail::SOAParams;
    using CSOAParams = generate_pipeline_detail::CSOAParams;

    GeneratePipeline();

    void record_device_code(CompileContext &cc, const Camera &camera, Film &film, FilmFilter &filter);

    void initialize(RC<cuda::Module> cuda_module, int spp, int state_count, const Vec2i &film_res);

    GeneratePipeline(GeneratePipeline &&other) noexcept;

    GeneratePipeline &operator=(GeneratePipeline &&other) noexcept;

    void swap(GeneratePipeline &other) noexcept;

    bool is_done() const;

    void clear();

    // returns number of new states
    int generate(
        int              active_state_count,
        const SOAParams &launch_params,
        int64_t          limit_max_state_count);

    float get_generated_percentage() const;

private:

    Vec2i   film_res_;
    int64_t pixel_count_;
    int64_t spp_;
    int64_t state_count_;

    int64_t finished_spp_;
    int64_t finished_pixel_;

    RC<cuda::Module> cuda_module_;
};

BTRC_WFPT_END
