#pragma once

#include <btrc/builtin/sampler/independent.h>
#include <btrc/utils/optix/pipeline_mk.h>

#define BTRC_PT_BEGIN BTRC_BUILTIN_BEGIN namespace pt {
#define BTRC_PT_END   } BTRC_BUILTIN_END

BTRC_PT_BEGIN

using GlobalSampler = IndependentSampler;

struct TraceUtils
{
    using Hit = optix::pipeline_mk_detail::Hit;

    std::function<Hit(const CRay &ray)>      find_closest_intersection;
    std::function<boolean(const CRay &ray)>  has_intersection;
};

BTRC_PT_END
