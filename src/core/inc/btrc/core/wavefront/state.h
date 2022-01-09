#pragma once

#include <btrc/core/wavefront/intersection.h>

BTRC_WAVEFRONT_BEGIN

/*
generate state: fill trace state
trace state: fill inct, splat le
bsdf state: sample light, generate shadow state, sample bsdf, fill trace state
shadow state: splat li
*/

/*
pipeline:
    while !done
        if active state count < threshold
            generate new state
        trace
        sort and modify active state count
        shade
        if active shadow state count > threshold
            shadow
*/

struct TraceState
{
    Vec2f           splat_coord;
    Ray             ray;
    SpectrumStorage beta;
    SpectrumStorage beta_le;
};

struct BSDFState
{
    Vec2f           splat_coord;
    Ray             ray;
    SpectrumStorage beta;
    Intersection    inct;
};

struct ShadowState
{
    Vec2f           splat_coord;
    Ray             ray;
    SpectrumStorage beta;
};

BTRC_WAVEFRONT_END
