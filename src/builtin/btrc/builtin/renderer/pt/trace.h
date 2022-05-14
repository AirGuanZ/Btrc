#pragma once

#include <btrc/builtin/renderer/pt/common.h>
#include <btrc/core/scene.h>
#include <btrc/utils/optix/pipeline_mk.h>

BTRC_PT_BEGIN

struct TraceResult
{
    CSpectrum radiance;
    CSpectrum albedo;
    CVec3f    normal;
};

struct TraceUtils
{
    using Hit = optix::pipeline_mk_detail::Hit;

    std::function<Hit(const CRay &ray)>      find_closest_intersection;
    std::function<boolean(const CRay &ray)>  has_intersection;
};

TraceResult trace_path(
    CompileContext   &cc,
    const TraceUtils &utils,
    const Params     &params,
    const Scene      &scene,
    const CRay       &ray,
    GlobalSampler    &sampler);

BTRC_PT_END
