#pragma once

#include <cuj.h>
#include <vector_types.h>

#include <btrc/core/common.h>

BTRC_WAVEFRONT_BEGIN

struct RaySOA
{
    float4 *o_t0;
    float4 *d_t1;
    uint2  *time_mask;
};

struct SpectrumSOA
{
    float *beta;
};

struct IntersectionSOA
{
    float *t;
    uint4 *uv_id;
};

struct RNGSOA
{
    cuj::cstd::LCGData *rng;
};

BTRC_WAVEFRONT_END
