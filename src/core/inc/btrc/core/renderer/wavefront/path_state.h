#pragma once

#include <btrc/core/spectrum/spectrum.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/cmath/cmath.h>

BTRC_WAVEFRONT_BEGIN

struct PathState
{
    void initialize(int state_count);

    void clear();

    void next_iteration();

    CUDABuffer<cstd::LCGData> rng;

    // ==== generate output ====

    // ray

    CUDABuffer<Vec4f> o_t0;
    CUDABuffer<Vec4f> d_t1;
    CUDABuffer<Vec2u> time_mask;

    // for direct illum with bsdf sampling

    CUDABuffer<Spectrum> beta_le;
    CUDABuffer<float>    bsdf_pdf;

    // path state

    CUDABuffer<Spectrum> beta;
    CUDABuffer<int32_t>  depth;
    CUDABuffer<Vec2f>    pixel_coord;
    CUDABuffer<Spectrum> path_radiance;

    // ==== trace output ====

    CUDABuffer<float>   inct_t;
    CUDABuffer<Vec4u>   inct_uv_id;
    CUDABuffer<int32_t> active_state_indices;

    // ==== shade output ====

    // 'state' state

    CUDABuffer<cstd::LCGData> next_rng;

    // path state

    CUDABuffer<Spectrum> next_beta;
    CUDABuffer<int32_t>  next_depth;
    CUDABuffer<Vec2f>    next_pixel_coord;
    CUDABuffer<Spectrum> next_path_radiance;

    // direct illum with light sampling

    CUDABuffer<Vec2f>    shadow_pixel_coord;
    CUDABuffer<Vec4f>    shadow_o_t0;
    CUDABuffer<Vec4f>    shadow_d_t1;
    CUDABuffer<Vec2u>    shadow_time_mask;
    CUDABuffer<Spectrum> shadow_beta_li;

    // next ray

    CUDABuffer<Vec4f> next_o_t0;
    CUDABuffer<Vec4f> next_d_t1;
    CUDABuffer<Vec2u> next_time_mask;

    // next direct illum with bsdf sampling

    CUDABuffer<Spectrum> next_beta_le;
    CUDABuffer<float>    next_bsdf_pdf;
};

BTRC_WAVEFRONT_END
