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

boolean sample_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    Sampler         &sampler,
    ref<CVec3f>      d,
    ref<f32>         t1,
    ref<f32>         light_pdf,
    ref<CSpectrum>   li);

boolean sample_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    const CVec3f    &ref_nor,
    Sampler         &sampler,
    ref<CVec3f>      o,
    ref<CVec3f>      d,
    ref<f32>         t1,
    ref<f32>         light_pdf,
    ref<CSpectrum>   li);

CSpectrum handle_miss(
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

BTRC_WFPT_END
