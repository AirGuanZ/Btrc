#pragma once

#include <btrc/core/camera/camera.h>
#include <btrc/core/utils/cuda/module.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_WAVEFRONT_BEGIN

namespace generate_pipeline_detail
{

    struct SOAParams
    {
        cstd::LCGData *rng;
        Vec2f         *output_pixel_coord;
        Vec4f         *output_ray_o_t0;
        Vec4f         *output_ray_d_t1;
        Vec2u         *output_ray_time_mask;
        float         *output_beta;
        float         *output_beta_le;
        float         *output_bsdf_pdf;
        int           *output_depth;
        float         *output_path_radiance;
    };

    CUJ_PROXY_CLASS(
        CSOAParams,
        SOAParams,
        rng,
        output_pixel_coord,
        output_ray_o_t0,
        output_ray_d_t1,
        output_ray_time_mask,
        output_beta,
        output_beta_le,
        output_bsdf_pdf,
        output_depth,
        output_path_radiance);

} // namespace generate_pipeline_detail

class GeneratePipeline : public Uncopyable
{
public:

    using SOAParams = generate_pipeline_detail::SOAParams;
    using CSOAParams = generate_pipeline_detail::CSOAParams;

    GeneratePipeline();

    explicit GeneratePipeline(
        const Camera &camera, const Vec2i &film_res, int spp, int state_count);

    GeneratePipeline(GeneratePipeline &&other) noexcept;

    GeneratePipeline &operator=(GeneratePipeline &&other) noexcept;

    void swap(GeneratePipeline &other) noexcept;

    operator bool() const;

    bool is_done() const;

    void clear();

    // returns number of new states
    int generate(
        int active_state_count,
        const SOAParams &launch_params);

private:

    void initialize(
        const Camera &camera, const Vec2i &film_res, int spp, int state_count);

    Vec2i film_res_;
    int   pixel_count_;
    int   spp_;
    int   state_count_;

    int finished_spp_;
    int finished_pixel_;

    CUDAModule kernel_;
};

BTRC_WAVEFRONT_END
