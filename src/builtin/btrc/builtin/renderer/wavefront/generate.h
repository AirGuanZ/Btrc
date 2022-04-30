#pragma once

#include <btrc/builtin/renderer/wavefront/soa.h>
#include <btrc/core/camera.h>
#include <btrc/core/film.h>
#include <btrc/core/film_filter.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace generate_pipeline_detail
{

    struct SOAParams
    {
        PathSOA   path;
        RaySOA    ray;
        BSDFLeSOA bsdf_le;
    };

    CUJ_PROXY_CLASS(
        CSOAParams,
        SOAParams,
        path,
        ray,
        bsdf_le);

} // namespace generate_pipeline_detail

class GeneratePipeline : public Uncopyable
{
public:

    enum class Mode
    {
        Uniform,
        Tile  
    };

    using SOAParams = generate_pipeline_detail::SOAParams;
    using CSOAParams = generate_pipeline_detail::CSOAParams;

    GeneratePipeline();

    void set_mode(Mode mode);

    void record_device_code(
        CompileContext &cc, const Scene &scene, const Camera &camera, Film &film, FilmFilter &filter, int spp);

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

    Mode mode_;

    Vec2i   film_res_;
    int64_t pixel_count_;
    int64_t spp_;
    int64_t state_count_;

    int64_t finished_spp_;
    int64_t finished_pixel_;

    RC<cuda::Module> cuda_module_;
};

BTRC_WFPT_END
