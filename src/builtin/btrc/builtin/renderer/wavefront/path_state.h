#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_WFPT_BEGIN

struct PathState
{
    void initialize(int state_count);

    void clear();

    void next_iteration();

    cuda::Buffer<CRNG::Data> rng;

    // ==== generate output ====

    // ray

    cuda::Buffer<Vec4f> o_medium_id;
    cuda::Buffer<Vec4f> d_t1;
    cuda::Buffer<Vec2u> time_mask;

    // for direct illum with bsdf sampling

    cuda::Buffer<Spectrum> beta_le;
    cuda::Buffer<float>    bsdf_pdf;

    // path state

    cuda::Buffer<Spectrum> beta;
    cuda::Buffer<int32_t>  depth;
    cuda::Buffer<Vec2u>    pixel_coord;
    cuda::Buffer<Spectrum> path_radiance;

    // ==== trace output ====

    cuda::Buffer<Vec2u> inct_inst_launch_index;
    cuda::Buffer<Vec4u> inct_t_prim_uv;

    // ==== shade output ====

    // 'state' state

    cuda::Buffer<CRNG::Data> next_rng;

    // path state

    cuda::Buffer<Spectrum> next_beta;
    cuda::Buffer<int32_t>  next_depth;
    cuda::Buffer<Vec2u>    next_pixel_coord;
    cuda::Buffer<Spectrum> next_path_radiance;

    // direct illum with light sampling

    cuda::Buffer<Vec2u>    shadow_pixel_coord;
    cuda::Buffer<Vec4f>    shadow_o_medium_id;
    cuda::Buffer<Vec4f>    shadow_d_t1;
    cuda::Buffer<Vec2u>    shadow_time_mask;
    cuda::Buffer<Spectrum> shadow_beta_li;

    // next ray

    cuda::Buffer<Vec4f>    next_o_medium_id;
    cuda::Buffer<Vec4f>    next_d_t1;
    cuda::Buffer<Vec2u>    next_time_mask;

    // next direct illum with bsdf sampling

    cuda::Buffer<Spectrum> next_beta_le;
    cuda::Buffer<float>    next_bsdf_pdf;

    // ==== shadow ====

    cuda::Buffer<CRNG::Data> shadow_rng;
};

BTRC_WFPT_END
