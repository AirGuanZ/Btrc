#pragma once

#include <cuj.h>

#include <btrc/common.h>

#define BTRC_WFPT_BEGIN BTRC_BUILTIN_BEGIN namespace wfpt {
#define BTRC_WFPT_END   } BTRC_BUILTIN_END

BTRC_WFPT_BEGIN

constexpr uint32_t INST_ID_MISS_MASK    = 0b01111111111111111111111111111111;
constexpr uint32_t INST_ID_SCATTER_MASK = 0b10000000000000000000000000000000;

inline cuj::boolean is_inct_miss(cuj::u32 inst_id)
{
    return (inst_id & INST_ID_MISS_MASK) == INST_ID_MISS_MASK;
}

inline cuj::boolean has_scattered(cuj::u32 inst_id)
{
    return (inst_id & INST_ID_SCATTER_MASK) != 0;
}

inline cuj::u32 get_raw_inst_id(cuj::u32 masked_inst_id)
{
    return masked_inst_id & ~INST_ID_SCATTER_MASK;
}

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
