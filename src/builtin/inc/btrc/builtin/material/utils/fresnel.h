#pragma once

#include <btrc/core/spectrum.h>

BTRC_BUILTIN_BEGIN

// eta_i: inner IOR. eta_o: eta_o outer IOR
f32 dielectric_fresnel(f32 eta_i, f32 eta_o, f32 cos_theta_i);

CSpectrum schlick_approx(ref<CSpectrum> R0, f32 cos_theta_i);

BTRC_BUILTIN_END
