#pragma once

#include <btrc/utils/cmath/cvec2.h>
#include <btrc/utils/cmath/cvec3.h>

BTRC_BEGIN

// uniform sphere

CVec3f sample_sphere_uniform(f32 u1, f32 u2);

CVec3f sample_sphere_uniform(ref<CVec2f> sam);

CVec3f sample_sphere_uniform(ref<cstd::LCG> rng);

f32 pdf_sample_sphere_uniform();

// z-weighted hemispherre

CVec3f sample_hemisphere_zweighted(f32 u1, f32 u2);

CVec3f sample_hemisphere_zweighted(ref<CVec2f> sam);

CVec3f sample_hemisphere_zweighted(ref<cstd::LCG> rng);

f32 pdf_sample_hemisphere_zweighted(ref<CVec3f> v); // v must be normalized

// uniform triangle

CVec2f sample_triangle_uniform(f32 u1, f32 u2);

BTRC_END
