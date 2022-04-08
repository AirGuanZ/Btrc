#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/core/scene.h>
#include <btrc/core/spectrum.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_WFPT_BEGIN

struct SimpleRussianRouletteParams
{
    int   min_depth      = 4;
    int   max_depth      = 8;
    float beta_threshold = 0.2f;
    float cont_prob      = 0.6f;
};

struct WFPTScene
{
    CompileContext &cc;
    const Scene    &scene;
    float           world_diagonal;
};

// returning true means 'exit'
boolean simple_russian_roulette(
    ref<CSpectrum>                     path_beta,
    i32                                depth,
    GlobalSampler                     &sampler,
    const SimpleRussianRouletteParams &params);

CSpectrum estimate_medium_tr(
    const WFPTScene &scene,
    CMediumID        medium_id,
    const CVec3f    &a,
    const CVec3f    &b,
    Sampler         &sampler);

struct SampleLiResult
{
    boolean   success;
    CSpectrum li;
    CVec3f    o;
    CVec3f    d;
    f32       t1;
    f32       light_pdf;
};

SampleLiResult sample_medium_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    Sampler         &sampler);

SampleLiResult sample_surface_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    const CVec3f    &ref_nor,
    Sampler         &sampler);

CSpectrum eval_miss_le(
    const WFPTScene &scene,
    const CVec3f    &o,
    const CVec3f    &d,
    const CSpectrum &beta_le,
    f32              bsdf_pdf);

CSpectrum handle_intersected_light(
    const WFPTScene    &scene,
    const CVec3f       &o,
    const CVec3f       &d,
    const SurfacePoint &inct,
    const CSpectrum    &beta_le,
    f32                 bsdf_pdf,
    i32                 light_index);

SurfacePoint get_intersection(
    const CVec3f &o,
    const CVec3f &d,
    const CInstanceInfo &instance,
    const CGeometryInfo &geometry,
    f32 t, u32 prim_id, const CVec2f &uv);

BTRC_WFPT_END
