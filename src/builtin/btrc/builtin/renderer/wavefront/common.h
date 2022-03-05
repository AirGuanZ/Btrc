#pragma once

#include <cuj.h>

#include <btrc/common.h>

#define BTRC_WFPT_BEGIN BTRC_BUILTIN_BEGIN namespace wfpt {
#define BTRC_WFPT_END   } BTRC_BUILTIN_END

BTRC_WFPT_BEGIN

using RNG = cuj::cstd::LCG;

constexpr uint32_t INST_ID_MISS        = static_cast<uint32_t>(-1);
constexpr uint32_t INST_ID_MEDIUM_MASK = 0x80000000;

struct StateCounters
{
    int32_t active_state_counter   = 0;
    int32_t inactive_state_counter = 0;
    int32_t shadow_ray_counter     = 0;
};

struct ShadeParams
{
    int   min_depth = 4;
    int   max_depth = 8;
    float rr_threshold = 0.2f;
    float rr_cont_prob = 0.6f;
};

BTRC_WFPT_END
