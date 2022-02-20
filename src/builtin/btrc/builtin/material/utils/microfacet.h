#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BUILTIN_BEGIN

namespace microfacet
{

    f32 gtr2(f32 cos_theta_h, f32 alpha);

    f32 smith_gtr2(f32 tan_theta, f32 alpha);

    CVec3f sample_gtr2(f32 alpha, ref<CVec2f> sample);

    f32 anisotropic_gtr2(f32 sin_phi_h, f32 cos_phi_h, f32 sin_theta_h, f32 cos_theta_h, f32 ax, f32 ay);

    f32 smith_anisotropic_gtr2(f32 cos_phi, f32 sin_phi, f32 ax, f32 ay, f32 tan_theta);

    CVec3f sample_anisotropic_gtr2(f32 ax, f32 ay, ref<CVec2f> sample);

    f32 gtr1(f32 sin_theta_h, f32 cos_theta_h, f32 alpha);

    CVec3f sample_gtr1(f32 alpha, ref<CVec2f> sample);

    CVec3f sample_anisotropic_gtr2_vnor(ref<CVec3f> ve, f32 ax, f32 ay, ref<CVec2f> sample);

} // namespace microfacet

BTRC_BUILTIN_END
