#pragma once

#include <btrc/builtin/sampler/independent.h>

#define BTRC_PT_BEGIN BTRC_BUILTIN_BEGIN namespace pt {
#define BTRC_PT_END   } BTRC_BUILTIN_END

BTRC_PT_BEGIN

using GlobalSampler = IndependentSampler;

struct Params
{
    int min_depth;
    int max_depth;

    float rr_threshold;
    float rr_cont_prob;

    bool albedo;
    bool normal;
};

BTRC_PT_END
