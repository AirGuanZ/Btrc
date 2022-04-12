#pragma once

#include <cuj.h>

#include <btrc/builtin/sampler/independent.h>

#define BTRC_WFPT_BEGIN BTRC_BUILTIN_BEGIN namespace wfpt {
#define BTRC_WFPT_END   } BTRC_BUILTIN_END

BTRC_WFPT_BEGIN

constexpr uint32_t PATH_FLAG_HAS_INTERSECTION = 0b01u << 30;
constexpr uint32_t PATH_FLAG_HAS_SCATTERING   = 0b10u << 30;
constexpr uint32_t PATH_FLAG_INSTANCE_ID_MASK = ~0u << 2 >> 2;

boolean is_path_intersected(u32 path_flag);

boolean is_path_scattered(u32 path_flag);

u32 extract_instance_id(u32 path_flag);

struct StateCounters
{
    int32_t active_state_counter = 0;
    int32_t shadow_ray_counter   = 0;
};

struct ShadeParams
{
    int   min_depth = 4;
    int   max_depth = 8;
    float rr_threshold = 0.2f;
    float rr_cont_prob = 0.6f;
};

//using GlobalSampler = IndependentSampler;
using GlobalSampler = IndependentSampler;

BTRC_WFPT_END
