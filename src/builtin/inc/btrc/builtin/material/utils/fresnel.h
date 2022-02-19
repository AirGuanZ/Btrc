#pragma once

#include <btrc/core/spectrum.h>

BTRC_BUILTIN_BEGIN

// eta_i: inner IOR. eta_o: eta_o outer IOR
f32 dielectric_fresnel(f32 eta_i, f32 eta_o, f32 cos_theta_i);

CSpectrum schlick_approx(ref<CSpectrum> R0, f32 cos_theta_i);

CUJ_CLASS_BEGIN(DielectricFresnelPoint)

    CUJ_MEMBER_VARIABLE(f32, eta_i)
    CUJ_MEMBER_VARIABLE(f32, eta_o)

    CSpectrum eval(f32 cos_theta_i) const
    {
        return CSpectrum::one() * dielectric_fresnel(eta_i, eta_o, cos_theta_i);
    }

CUJ_CLASS_END

CUJ_CLASS_BEGIN(ConductorFresnelPoint)

    CUJ_MEMBER_VARIABLE(CSpectrum, R0)

    CSpectrum eval(f32 cos_theta_i) const
    {
        return schlick_approx(R0, cos_theta_i);
    }

CUJ_CLASS_END

BTRC_BUILTIN_END
