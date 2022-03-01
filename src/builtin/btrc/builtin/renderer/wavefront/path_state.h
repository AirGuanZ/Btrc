#pragma once

#include <btrc/core/spectrum.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/cmath/cmath.h>

#include "./common.h"

BTRC_WFPT_BEGIN

struct PathState
{
    void initialize(int state_count);

    void clear();

    void next_iteration();

    cuda::CUDABuffer<RNG::Data> rng;

    // ==== generate output ====

    // ray

    cuda::CUDABuffer<Vec4f> o_t0;
    cuda::CUDABuffer<Vec4f> d_t1;
    cuda::CUDABuffer<Vec2u> time_mask;

    // for direct illum with bsdf sampling

    cuda::CUDABuffer<Spectrum> beta_le;
    cuda::CUDABuffer<float>    bsdf_pdf;

    // path state

    cuda::CUDABuffer<Spectrum> beta;
    cuda::CUDABuffer<int32_t>  depth;
    cuda::CUDABuffer<Vec2f>    pixel_coord;
    cuda::CUDABuffer<Spectrum> path_radiance;

    // ==== trace output ====

    cuda::CUDABuffer<Vec2u> inct_inst_launch_index;
    cuda::CUDABuffer<Vec4u> inct_t_prim_uv;

    // ==== shade output ====

    // 'state' state

    cuda::CUDABuffer<RNG::Data> next_rng;

    // path state

    cuda::CUDABuffer<Spectrum> next_beta;
    cuda::CUDABuffer<int32_t>  next_depth;
    cuda::CUDABuffer<Vec2f>    next_pixel_coord;
    cuda::CUDABuffer<Spectrum> next_path_radiance;

    // direct illum with light sampling

    cuda::CUDABuffer<Vec2f>    shadow_pixel_coord;
    cuda::CUDABuffer<Vec4f>    shadow_o_t0;
    cuda::CUDABuffer<Vec4f>    shadow_d_t1;
    cuda::CUDABuffer<Vec2u>    shadow_time_mask;
    cuda::CUDABuffer<Spectrum> shadow_beta_li;

    // next ray

    cuda::CUDABuffer<Vec4f> next_o_t0;
    cuda::CUDABuffer<Vec4f> next_d_t1;
    cuda::CUDABuffer<Vec2u> next_time_mask;

    // next direct illum with bsdf sampling

    cuda::CUDABuffer<Spectrum> next_beta_le;
    cuda::CUDABuffer<float>    next_bsdf_pdf;
};

BTRC_WFPT_END
