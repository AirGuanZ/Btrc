#pragma once

#include <btrc/builtin/renderer/wavefront/volume.h>
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

// returning true means 'exit'
boolean simple_russian_roulette(
    ref<CSpectrum>                     path_beta,
    i32                                depth,
    GlobalSampler                     &sampler,
    const SimpleRussianRouletteParams &params);

CSpectrum estimate_medium_tr(
    CompileContext      &cc,
    const Scene         &scene,
    const VolumeManager &vols,
    CMediumID            medium_id,
    const CVec3f        &a,
    const CVec3f        &b,
    GlobalSampler       &sampler);

BTRC_WFPT_END
