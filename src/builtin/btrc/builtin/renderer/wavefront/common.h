#pragma once

#include <cuj.h>

#include <btrc/common.h>

#define BTRC_WFPT_BEGIN BTRC_BUILTIN_BEGIN namespace wfpt {
#define BTRC_WFPT_END   } BTRC_BUILTIN_END

BTRC_WFPT_BEGIN

constexpr uint32_t PATH_FLAG_ACTIVE           = 0b001u << 29;
constexpr uint32_t PATH_FLAG_HAS_INTERSECTION = 0b010u << 29;
constexpr uint32_t PATH_FLAG_HAS_SCATTERING   = 0b100u << 29;
constexpr uint32_t PATH_FLAG_INSTANCE_ID_MASK = ~0u << 3 >> 3;

cuj::boolean is_path_active(cuj::u32 path_state);

cuj::boolean is_path_intersected(cuj::u32 path_state);

cuj::boolean is_path_scattered(cuj::u32 path_state);

cuj::u32 extract_instance_id(cuj::u32 path_state);

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
