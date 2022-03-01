#pragma once

#include <cuj.h>

#include <btrc/common.h>

#define BTRC_WFPT_BEGIN BTRC_BUILTIN_BEGIN namespace wfpt {
#define BTRC_WFPT_END   } BTRC_BUILTIN_END

BTRC_WFPT_BEGIN

using RNG = cuj::cstd::LCG;

BTRC_WFPT_END
