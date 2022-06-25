#pragma once

#include <btrc/builtin/renderer/pt/pm.h>

BTRC_PT_BEGIN

struct TraceLightPathParams
{
    int min_depth;
    int max_depth;

    float rr_threshold;
    float rr_cont_prob;
};

struct WorldBound
{
    // excluding camera

    Vec3f scene_center;
    float scene_radius;

    // including camera

    Vec3f world_center;
    float world_radius;
};

void trace_light_path(
    CompileContext             &cc,
    const TraceUtils           &utils,
    const TraceLightPathParams &params,
    const Scene                &scene,
    GlobalSampler              &sampler,
    PhotonMap                  &photon_map,
    const WorldBound           &world_bound);

BTRC_PT_END
