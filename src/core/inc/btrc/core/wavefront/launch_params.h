#pragma once

#include <cstdint>

#include <optix.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <btrc/core/utils/bitcast.h>

BTRC_WAVEFRONT_BEGIN

struct GenerateOutput
{
    // ray

    float4 *ray_o_t0;
    float4 *ray_d_t1;
    uint2  *ray_time_mask;

    // throughput and filter weight

    float4 *throughput_weight;
};

struct TraceParams
{
    OptixTraversableHandle handle;
    
    // ray

    const float4 *ray_o_t0;
    const float4 *ray_d_t1;
    const uint2  *ray_time_mask;

    // incts

    float *inct_t;
    uint4 *inct_uv_id;
};

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void load_ray(
    const T  &params,
    uint32_t  idx,
    float3   &o,
    float3   &d,
    float    &t0,
    float    &t1,
    float    &time,
    uint32_t &mask);

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_ray(
    const T      &params,
    uint32_t      idx,
    const float3 &o,
    const float3 &d,
    float         t0,
    float         t1,
    float         time,
    uint32_t      mask);

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_invalid_inct(
    const T  &params,
    uint32_t  idx);

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_inct(
    const T      &params,
    uint32_t      idx,
    float         t,
    const float2 &uv,
    uint32_t      prim_id,
    uint32_t      inst_id);

// ========================== impl ==========================

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void load_ray(
    const T  &params,
    uint32_t  idx,
    float3   &o,
    float3   &d,
    float    &t0,
    float    &t1,
    float    &time,
    uint32_t &mask)
{
    const float4 o_t0 = params.ray_o_t0[idx];
    const float4 d_t1 = params.ray_d_t1[idx];
    const uint2 time_mask = params.ray_time_mask[idx];

    o = make_float3(o_t0.x, o_t0.y, o_t0.z);
    d = make_float3(d_t1.x, d_t1.y, d_t1.z);
    t0 = o_t0.w;
    t1 = d_t1.w;
    time = bitcast<float>(time_mask.x);
    mask = time_mask.y;
}

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_ray(
    const T      &params,
    uint32_t      idx,
    const float3 &o,
    const float3 &d,
    float         t0,
    float         t1,
    float         time,
    uint32_t      mask)
{
    params.ray_o_t0[idx] = make_float4(o.x, o.y, o.z, t0);
    params.ray_d_t1[idx] = make_float4(d.x, d.y, d.z, t1);
    params.ray_time_mask[idx] = make_uint2(bitcast<uint32_t>(time), mask);
}

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_invalid_inct(
    const T  &params,
    uint32_t  idx)
{
    params.inct_t[idx] = -1;
}

template<typename T>
BTRC_XPU BTRC_FORCEINLINE void save_inct(
    const T      &params,
    uint32_t      idx,
    float         t,
    const float2 &uv,
    uint32_t      prim_id,
    uint32_t      inst_id)
{
    params.inct_t[idx] = t;
    params.inct_uv_id[idx] = make_uint4(
        bitcast<uint32_t>(uv.x),
        bitcast<uint32_t>(uv.y),
        prim_id, inst_id);
}

BTRC_WAVEFRONT_END
