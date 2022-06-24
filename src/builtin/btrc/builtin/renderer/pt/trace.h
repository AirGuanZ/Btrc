#pragma once

#include <btrc/builtin/renderer/pt/common.h>
#include <btrc/core/scene.h>

BTRC_PT_BEGIN

struct TraceParams
{
    int min_depth;
    int max_depth;

    float rr_threshold;
    float rr_cont_prob;

    bool albedo;
    bool normal;
};

struct TraceResult
{
    CSpectrum radiance;
    CSpectrum albedo;
    CVec3f    normal;
};

TraceResult trace_path(
    CompileContext    &cc,
    const TraceUtils  &utils,
    const TraceParams &params,
    const Scene       &scene,
    const CRay        &ray,
    CMediumID          initial_ray_medium_id,
    GlobalSampler     &sampler,
    float              world_diagonal);

BTRC_PT_END
