#pragma once

#include <btrc/builtin/sampler/independent.h>

#define BTRC_LPM_BEGIN BTRC_BUILTIN_BEGIN namespace lpm {
#define BTRC_LPM_END   } BTRC_BUILTIN_END

BTRC_LPM_BEGIN

using GlobalSampler = IndependentSampler;

BTRC_LPM_END
