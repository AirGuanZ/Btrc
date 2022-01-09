#pragma once

#include <btrc/core/camera/camera.h>
#include <btrc/core/utils/cuda/module.h>
#include <btrc/core/utils/uncopyable.h>
#include <btrc/core/wavefront/launch_params.h>

BTRC_WAVEFRONT_BEGIN

class GeneratePipeline : public Uncopyable
{
public:

    GeneratePipeline();

    explicit GeneratePipeline(
        const Camera &camera, const Vec2i &film_res, int spp, int state_count);

    GeneratePipeline(GeneratePipeline &&other) noexcept;

    GeneratePipeline &operator=(GeneratePipeline &&other) noexcept;

    void swap(GeneratePipeline &other) noexcept;

    operator bool() const;

    bool is_done() const;

    // returns number of new states
    int generate(
        int            active_state_count,
        cstd::LCGData *rngs,
        float2        *output_pixel_coord,
        float4        *output_ray_o_t0,
        float4        *output_ray_o_t1,
        uint2         *output_ray_time_mask);

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
