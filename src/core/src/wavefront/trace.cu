#include <cassert>
#include <cstdlib>

#include <vector_functions.h>

#include <btrc/core/wavefront/launch_params.h>

using namespace btrc::core;

extern "C"
{
    __constant__ wf::TraceParams launch_params;
}

extern "C" __global__ void __raygen__trace()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    assert(launch_idx.z == 0 && launch_dim.z == 1);
    const uint32_t idx = launch_idx.y * launch_dim.x + launch_idx.x;
    
    float3 o, d; float t0, t1, time; uint32_t mask;
    wf::load_ray(launch_params, idx, o, d, t0, t1, time, mask);
    assert(0 <= t0 && t0 <= t1);
    
    uint32_t payload0 = idx;
    optixTrace(
        launch_params.handle,
        o, d, t0, t1, time, mask,
        OPTIX_RAY_FLAG_NONE, 0, 1, 0, payload0);
}

extern "C" __global__ void __miss__trace()
{
    const uint32_t idx = optixGetPayload_0();
    wf::save_invalid_inct(launch_params, idx);
}

extern "C" __global__ void __closesthit__trace()
{
    const uint32_t idx = optixGetPayload_0();
    const float t = optixGetRayTmax();
    const float2 uv = optixGetTriangleBarycentrics();
    const uint32_t prim_id = optixGetPrimitiveIndex();
    const uint32_t inst_id = optixGetInstanceId();
    wf::save_inct(launch_params, idx, t, uv, prim_id, inst_id);
}
